import csv
import math
import urllib.request
from zipfile import ZipFile

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

import gzip
import json

from io import BytesIO


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
    """Parses lines from json file."""
    with gzip.open(path, "rt") as gz:
        for line in gz:
            yield json.loads(
                line, object_hook=lambda dct: tuple(dct.get(key, dct) for key in fields)
            )


def get_reviews(data_file):
    """Converts raw data to csv."""
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
        zip_file_url = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
        with urllib.request.urlopen(zip_file_url) as zip_response:
            zip_contents = BytesIO(zip_response.read())
        print("downloaded")
    else:
        zip_contents = local_file
    ml_data = ml_genres = ml_tags = mapping = None
    # loading data into memory
    with ZipFile(zip_contents) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains("rating")].iat[0]
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
    var_name,
    dataset_matrix,
    num_vals_batch_size,
    num_vals_rank,
    num_tries=None,
    min_rank=None,
    min_batch=None,
    max_rank=None,
    max_batch=None,
    percents=True,
):
    """Builds lists of points, where estimations should be made."""
    if not max_rank or max_rank > min(dataset_matrix.shape):
        max_rank = min(dataset_matrix.shape)
    if not min_rank or min_rank > min(dataset_matrix.shape):
        min_rank = 2

    if not max_batch or max_batch > max(dataset_matrix.shape):
        max_batch = max(dataset_matrix.shape)
    if not min_batch or min_batch > max(dataset_matrix.shape):
        min_batch = 2

    if var_name == "Rank":
        # abs_min_rank = min(dataset_matrix.shape) * min_rank // 100
        # abs_max_rank = min(dataset_matrix.shape) * max_rank // 100
        abs_min_rank = min_rank
        abs_max_rank = max_rank
        return np.linspace(abs_min_rank, abs_max_rank, num_vals_rank, dtype=int)
        # return [32, 64, 128, 256, 512, 1024, 1536, 2048, 3072]

        # return [1000]
    if var_name == "Batch_size":
        if percents:
            print(dataset_matrix.shape)
            abs_min_batch = dataset_matrix.shape[1] * min_batch // 100
            abs_max_batch = dataset_matrix.shape[1] * max_batch // 100
            # abs_min_batch = min_batch
            # abs_max_batch = max_batch
        else:
            abs_min_batch = min_batch
            abs_max_batch = max_batch
            # print(int(math.log2(abs_max_batch)))

        # return [
        #     2**i
        #     for i in range(
        #         int(math.log2(abs_min_batch)),
        #         int(np.round(math.log2(abs_max_batch))) + 1,
        #         1,
        #     )
        # ]
        return np.linspace(abs_min_batch, abs_max_batch, num_vals_batch_size, dtype=int)
        # return [3617, 7234, 10851, 14468, 18085]
    if var_name == "N_tries":
        return [25]
    if var_name == "Way":
        return ["top_rand", "new"]


def add_data(csv_path, dict_vals):
    """Adds data to csv file."""
    new_rows = []
    new_rows.append(dict_vals.values())
    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)
