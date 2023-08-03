import csv
from zipfile import ZipFile

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

import gzip
import json

from io import BytesIO

try:
    import networkx as nx
except ImportError:
    nx = None


def reindex(raw_data, index, filter_invalid=True, names=None):
    """
    Factorizes column values based on provided pandas index. Allows resetting
    index names. Optionally drops rows with entries not present in the index.
    """
    if isinstance(index, pd.Index):
        index = [index]
    if isinstance(names, str):
        names = [names]
    if isinstance(names, (list, tuple, pd.Index)):
        for i, name in enumerate(names):
            index[i].name = name
    new_data = raw_data.assign(
        **{idx.name: idx.get_indexer(raw_data[idx.name]) for idx in index}
    )

    if filter_invalid:
        # pandas returns -1 if label is not present in the index
        # checking if -1 is present anywhere in data
        maybe_invalid = new_data.eval(
            " or ".join([f"{idx.name} == -1" for idx in index])
        )
        if maybe_invalid.any():
            print(f"Filtered {maybe_invalid.sum()} invalid observations.")
            new_data = new_data.loc[~maybe_invalid]
    return


def matrix_from_observations(
    data,
    userid="userid",
    itemid="itemid",
    user_index=None,
    item_index=None,
    feedback=None,
    preserve_order=False,
    shape=None,
    dtype=None,
):
    """
    Encodes pandas dataframe into sparse matrix. If index is not provided,
    returns new index mapping, which optionally preserves order of original data.
    Automatically removes incosnistent data not present in the provided index.
    """
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
    with gzip.open(path, "rt") as gz:
        for line in gz:
            yield json.loads(
                line, object_hook=lambda dct: tuple(dct.get(key, dct) for key in fields)
            )


def get_reviews(data_file):
    fields = ["reviewerID", "asin"]
    lines = parse_lines(data_file, fields)
    pcore_data = pd.DataFrame.from_records(
        parse_lines(data_file, fields),
        columns=fields,
    ).rename(columns={"reviewerID": "userid", "asin": "itemid"})
    return pcore_data


def get_movielens_data(
    local_file=None,
    get_ratings=True,
    get_genres=False,
    split_genres=True,
    mdb_mapping=False,
    get_tags=False,
    include_time=False,
):
    """Downloads movielens data and stores it in pandas dataframe."""
    fields = ["userid", "movieid", "rating"]

    if include_time:
        fields.append("timestamp")
    if not local_file:
        # downloading data
        zip_file_url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
        with urllib.request.urlopen(zip_file_url) as zip_response:
            zip_contents = BytesIO(zip_response.read())
    else:
        zip_contents = local_file
    ml_data = ml_genres = ml_tags = mapping = None
    # loading data into memory
    with ZipFile(zip_contents) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains("ratings")].iat[0]
        is_new_format = (
            ("latest" in zip_file) or ("20m" in zip_file) or ("25m" in zip_file)
        )
        delimiter = ","
        header = 0 if is_new_format else None
        if get_ratings:
            zdata = zfile.read(zip_file)
            zdata = zdata.replace(b"::", delimiter.encode())
            # makes data compatible with pandas c-engine
            # returns string objects instead of bytes in that case
            ml_data = pd.read_csv(
                BytesIO(zdata),
                sep=delimiter,
                header=header,
                engine="c",
                names=fields,
                usecols=fields,
            )

        if get_tags:
            zip_file = zip_files[zip_files.str.contains("/tags")].iat[0]  # not genome
            zdata = zfile.read(zip_file)
            if not is_new_format:
                # make data compatible with pandas c-engine
                # pandas returns string objects instead of bytes in that case
                delimiter = "^"
                zdata = zdata.replace(b"::", delimiter.encode())
            fields[2] = "tag"
            ml_tags = pd.read_csv(
                BytesIO(zdata),
                sep=delimiter,
                header=header,
                engine="c",
                encoding="latin1",
                names=fields,
                usecols=range(len(fields)),
            )
        if mdb_mapping and is_new_format:
            # imdb and tmdb mapping - exists only in ml-latest or 20m datasets
            zip_file = zip_files[zip_files.str.contains("links")].iat[0]
            with zfile.open(zip_file) as zdata:
                mapping = pd.read_csv(
                    zdata,
                    sep=",",
                    header=0,
                    engine="c",
                    names=["movieid", "imdbid", "tmdbid"],
                )
    res = [data for data in [ml_data, ml_genres, ml_tags, mapping] if data is not None]
    return res[0] if len(res) == 1 else res


def make_list_params(
    var_name, dataset_matrix, num_vals, up_bound, max_rank=None, max_batch=None
):
    if not max_rank:
        max_rank = min(dataset_matrix.shape)
    if not max_batch:
        max_batch = max(dataset_matrix.shape)
    if var_name == "Rank":
        return [
            j
            for j in range(
                max_rank // num_vals,
                max_rank + max_rank // num_vals,
                max_rank // num_vals,
            )
        ][:up_bound]
        # return [1000]
    if var_name == "Batch_size":
        return [
            j
            for j in range(
                max_batch // num_vals,
                max_batch + max_batch // num_vals,
                max_batch // num_vals,
            )
        ][:up_bound]
    if var_name == "N_tries":
        return range(10, 60, 10)
    if var_name == "Way":
        return ["new", "old"]


def add_data(
    deltas_diams, time, k, b_s, n_try, way, dataset, csv_path, svd_Time, mult_Time
):
    new_rows = []
    deltas = list(map(lambda x: x[0], deltas_diams))
    diams = list(map(lambda x: x[1], deltas_diams))
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas)
    mean_diam = np.mean(diams)
    std_diam = np.std(diams)
    for l, delta in enumerate(deltas):
        new_rows.append(
            [
                delta,
                diams[l],
                dataset,
                mean_delta,
                std_delta,
                k,
                b_s,
                n_try,
                mean_diam,
                std_diam,
                time,
                svd_Time,
                mult_Time,
                way,
            ]
        )
    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)
