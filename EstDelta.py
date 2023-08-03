import argparse
import csv
from os import listdir
from os.path import isfile, join
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix

from scipy.sparse.linalg import svds, aslinearoperator

from delta import batched_delta_hyp_old, batched_delta_hyp
import gzip
import json
import time
import os

from io import BytesIO

try:
    import networkx as nx
except ImportError:
    nx = None
# import polara


def reindex(raw_data, index, filter_invalid=True, names=None):
    '''
    Factorizes column values based on provided pandas index. Allows resetting
    index names. Optionally drops rows with entries not present in the index.
    '''
    if isinstance(index, pd.Index):
        index = [index]

    if isinstance(names, str):
        names = [names]

    if isinstance(names, (list, tuple, pd.Index)):
        for i, name in enumerate(names):
            index[i].name = name

    new_data = raw_data.assign(**{
        idx.name: idx.get_indexer(raw_data[idx.name]) for idx in index
    })

    if filter_invalid:
        # pandas returns -1 if label is not present in the index
        # checking if -1 is present anywhere in data
        maybe_invalid = new_data.eval(
            ' or '.join([f'{idx.name} == -1' for idx in index])
        )
        if maybe_invalid.any():
            print(f'Filtered {maybe_invalid.sum()} invalid observations.')
            new_data = new_data.loc[~maybe_invalid]

    return

#
def matrix_from_observations(
        data,
        userid='userid',
        itemid='itemid',
        user_index=None,
        item_index=None,
        feedback=None,
        preserve_order=False,
        shape=None,
        dtype=None
    ):
    '''
    Encodes pandas dataframe into sparse matrix. If index is not provided,
    returns new index mapping, which optionally preserves order of original data.
    Automatically removes incosnistent data not present in the provided index.
    '''
    if (user_index is None) or (item_index is None):
        useridx, user_index = pd.factorize(data[userid], sort=preserve_order)
        itemidx, item_index = pd.factorize(data[itemid], sort=preserve_order)
        user_index.name = userid
        item_index.name = itemid
    else:
        data = reindex(data, (user_index, item_index), filter_invalid=True)
        useridx = data[userid].values
        itemidx = data[itemid].values
        if shape is None:
            shape = (len(user_index), len(item_index))

    if feedback is None:
        values = np.ones_like(itemidx, dtype=dtype)
    else:
        values = data[feedback].values

    matrix = csr_matrix((values, (useridx, itemidx)), dtype=dtype, shape=shape)
    return matrix, user_index, item_index


def parse_lines(path, fields):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield json.loads(line, object_hook=lambda dct: tuple(dct.get(key, dct) for key in fields))


def get_reviews(data_file):
    fields = ['reviewerID', 'asin']
    lines = parse_lines(data_file, fields)
    pcore_data = pd.DataFrame.from_records(
        parse_lines(data_file, fields),
        columns=fields,
    ).rename(columns={'reviewerID': 'userid', 'asin':'itemid'})
    return pcore_data


def get_movielens_data(local_file=None, get_ratings=True, get_genres=False,
                       split_genres=True, mdb_mapping=False, get_tags=False, include_time=False):
    '''Downloads movielens data and stores it in pandas dataframe.
    '''
    fields = ['userid', 'movieid', 'rating']

    if include_time:
        fields.append('timestamp')

    if not local_file:
        # downloading data
        zip_file_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        with urllib.request.urlopen(zip_file_url) as zip_response:
            zip_contents = BytesIO(zip_response.read())
    else:
        zip_contents = local_file

    ml_data = ml_genres = ml_tags = mapping = None
    # loading data into memory
    with ZipFile(zip_contents) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains('ratings')].iat[0]
        is_new_format = ('latest' in zip_file) or ('20m' in zip_file) or ('25m' in zip_file)
        delimiter = ','
        header = 0 if is_new_format else None
        if get_ratings:
            zdata = zfile.read(zip_file)
            zdata = zdata.replace(b'::', delimiter.encode())
            # makes data compatible with pandas c-engine
            # returns string objects instead of bytes in that case
            ml_data = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header, engine='c', names=fields, usecols=fields)

        # if get_genres:
        #     zip_file = zip_files[zip_files.str.contains('movies')].iat[0]
        #     zdata =  zfile.read(zip_file)
        #     if not is_new_format:
        #         # make data compatible with pandas c-engine
        #         # pandas returns string objects instead of bytes in that case
        #         delimiter = '^'
        #         zdata = zdata.replace(b'::', delimiter.encode())
        #     genres_data = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header,
        #                               engine='c', encoding='unicode_escape',
        #                               names=['movieid', 'movienm', 'genres'])
        #
        #     ml_genres = get_split_genres(genres_data) if split_genres else genres_data

        if get_tags:
            zip_file = zip_files[zip_files.str.contains('/tags')].iat[0] #not genome
            zdata =  zfile.read(zip_file)
            if not is_new_format:
                # make data compatible with pandas c-engine
                # pandas returns string objects instead of bytes in that case
                delimiter = '^'
                zdata = zdata.replace(b'::', delimiter.encode())
            fields[2] = 'tag'
            ml_tags = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header,
                                      engine='c', encoding='latin1',
                                      names=fields, usecols=range(len(fields)))

        if mdb_mapping and is_new_format:
            # imdb and tmdb mapping - exists only in ml-latest or 20m datasets
            zip_file = zip_files[zip_files.str.contains('links')].iat[0]
            with zfile.open(zip_file) as zdata:
                mapping = pd.read_csv(zdata, sep=',', header=0, engine='c',
                                        names=['movieid', 'imdbid', 'tmdbid'])

    res = [data for data in [ml_data, ml_genres, ml_tags, mapping] if data is not None]
    return res[0] if len(res)==1 else res


def make_list(var_name, dataset_matrix, num_vals, up_bound, max_rank=None, max_batch=None):
    if not max_rank:
        max_rank = min(dataset_matrix)
    if not max_batch:
        max_batch = max(dataset_matrix)
    if var_name == 'Rank':
        return [j for j in range(max_rank // num_vals, max_rank + max_rank // num_vals, max_rank // num_vals)][:up_bound]
        # return [1000]
    if var_name == 'Batch_size':
        return [j for j in range(max_batch // num_vals, max_batch + max_batch // num_vals, max_batch // num_vals)][:up_bound]
    if var_name == 'N_tries':
        return range(10, 60, 10)
    if var_name == 'Way':
        return ['new', 'old']


def build_csv(datasets_dir, csv_dir, max_batch=0, max_rank=0, num_vals=10, up_bound=10, first_dependency='Rank', second_dependency='Batch_size',
              third_dependency=None):
    if third_dependency:
        csv_file_name = f'try_{first_dependency}_{second_dependency}_{third_dependency}.csv'
    else:
        csv_file_name = f'{first_dependency}_{second_dependency}.csv'
    datafiles = [f for f in listdir(datasets_dir) if isfile(join(datasets_dir, f))]
    df = pd.DataFrame(
        columns=['Delta', 'Diam', 'Dataset', 'Mean_delta', 'Std_delta', 'Rank', 'Batch_size', 'Num_of_attempts',
                 'Mean_diam', 'Std_diam', 'all_Time',  'svd_Time', 'mult_Time', 'Way'])
    path_to_csv = os.path.join(csv_dir, csv_file_name)
    df.to_csv(path_to_csv, index=False)

    datafiles.sort()

    val_list_dict = {
        'Rank': [50],
        'Batch_size': [500],
        'Way': ['old'],
        'N_tries': [5]
    }

    for i, datafile in enumerate(datafiles):
        if datafile[-2:] == 'gz':
            print(datafile[:-10])
        elif datafile[-3:] == 'zip':
            print(datafile[:-4])
        if datafile[:-4] == 'Movielens_1m':
            cur_df = get_movielens_data(join(datasets_dir, datafile))
            matr_from_observ, u_id, i_id = matrix_from_observations(cur_df,  dtype=float, itemid='movieid')
        else:
            cur_df = get_reviews(join(datasets_dir, datafile))
            matr_from_observ, u_id, i_id = matrix_from_observations(cur_df,  dtype=float)

        if max_batch == 0:
            max_batch = max(matr_from_observ.shape)
        if max_rank == 0:
            max_rank = min(matr_from_observ.shape)

        val_list_dict[first_dependency] = make_list(first_dependency, matr_from_observ, num_vals, up_bound, max_rank=max_rank, max_batch=max_batch)
        val_list_dict[second_dependency] = make_list(second_dependency, matr_from_observ, num_vals, up_bound, max_rank=max_rank, max_batch=max_batch)
        if third_dependency:
            val_list_dict[third_dependency] = make_list(third_dependency, matr_from_observ, num_vals, up_bound, max_rank=max_rank, max_batch=max_batch)
        for way in val_list_dict['Way']:
            if way == 'old':
                st_1 = time.time()
                U, S, V = randomized_svd(matr_from_observ, n_components=max_rank)

                indices = np.flip(np.argsort(S))
                new_S = [S[i] for i in indices]
                et_1 = time.time()
                print('done svd, time: ' + str(et_1 - st_1))

                for l, k in enumerate(val_list_dict['Rank']):
                    st_2 = time.time()
                    item_space = V.T[:, indices[:k]] @ np.diag(new_S[:k])
                    et_2 = time.time()
                    for j, b_s in enumerate(val_list_dict['Batch_size']):
                        # cur_batch = (j + 1) * b_s_part
                        # cur_b_s = b_s // max(matr_from_observ.shape)
                        for n_try in val_list_dict['N_tries']:
                            # cur_rank = (l + 1) * rank_part
                            st_old = time.time()
                            deltas_diams = batched_delta_hyp_old(item_space, economic=True, batch_size=b_s, n_tries=n_try, seed=42)
                            et_old = time.time()
                            add_data(deltas_diams=deltas_diams,
                                     time=(et_old - st_old) + (et_1 - st_1) + (et_2 - st_2), k=k,
                                     mult_Time=(et_2 - st_2), svd_Time=(et_1 - st_1),
                                     b_s=(b_s * 100) // max_batch, n_try=n_try, way=way, dataset=datafile[:-10],
                                     csv_path=path_to_csv)
                            print('done try ' + str(n_try))
                        print('done batch_size ' + str(b_s))
                    print('done rank ' + str(k))
                print('done ' + str(way))
            if way == 'new':
                # work_matrix = csr_matrix.toarray(matr_from_observ)
                for l, k in enumerate(val_list_dict['Rank']):
                    for j, b_s in enumerate(val_list_dict['Batch_size']):
                        for n_try in val_list_dict['N_tries']:
                            #
                            st = time.time()
                            deltas_diams = batched_delta_hyp(matr_from_observ, economic=True, rank=k, batch_size=b_s,
                                                             n_tries=n_try, seed=42)
                            et = time.time()
                            add_data(deltas_diams=deltas_diams, time=et - st, svd_Time=0, mult_Time=0,
                                     k=k, b_s=(b_s * 100)//max_batch, n_try=n_try, way=way,
                                     dataset=datafile[:-10], csv_path=path_to_csv)
                            print('done try ' + str(n_try))
                        print('done batch_size ' + str((b_s * 100)//max_batch) + ' %')
                    print('done rank ' + str(k))
                print('done ' + str(way))


def add_data(deltas_diams, time, k, b_s, n_try, way, dataset, csv_path, svd_Time, mult_Time):
    new_rows = []
    deltas = list(map(lambda x: x[0], deltas_diams))
    diams = list(map(lambda x: x[1], deltas_diams))
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    mean_diam = np.mean(diams)
    std_diam = np.std(diams)
    for l, delta in enumerate(deltas):
        new_rows.append([delta, diams[l], dataset, mean_delta, std_delta, k, b_s, n_try,
                 mean_diam, std_diam, time, svd_Time, mult_Time, way])
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)


def main(datasets_dir, csv_dir, path_to_config, max_batch, max_rank, grid_size, up_bound):
    with open(path_to_config, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    datasets_types = data["DATASETS"]
    cur_datasets = datasets_types['current']

    dependencies = [('Batch_size', 'Rank', 'Way')]
    for dependency in dependencies:
        build_csv(
            datasets_dir, csv_dir, max_batch, max_rank, num_vals=grid_size, up_bound=up_bound,
            first_dependency=dependency[0],
            second_dependency=dependency[1],
            third_dependency='Way'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Builds csv files"
    )
    parser.add_argument(
        "datasets_dir", type=str, help="Directory where datasets are stored"
    )
    parser.add_argument(
        "csv_dir", type=str, help="Directory where csvs are stored"
    )
    parser.add_argument(
        "config_path", type=str, help="Path ro config file"
    )
    parser.add_argument(
        "max_batch", type=int, help="Maximum batch size"
    )
    parser.add_argument(
        "max_rank", type=int, help="Maximum rank"
    )
    parser.add_argument(
        "grid_size", type=int, help="num points on each axis"
    )
    parser.add_argument(
        "up_bound", type=int, help="num factual points on each axis"
    )

    args = parser.parse_args()
    main(
        args.datasets_dir,
        args.csv_dir,
        args.config_path,
        args.max_batch,
        args.max_rank,
        args.grid_size,
        args.up_bound
    )
