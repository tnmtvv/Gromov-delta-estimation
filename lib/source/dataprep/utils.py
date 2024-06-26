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
from timeit import default_timer as timer


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


def get_reviews(data_file, include_time=False):
    """Converts raw data to csv."""
    if not include_time:
        fields = ["reviewerID", "asin"]
    else:
        fields = ["reviewerID", "asin", "unixReviewTime"]

    pcore_data = pd.DataFrame.from_records(
        parse_lines(data_file, fields),
        columns=fields,
    )

    if include_time:
        pcore_data.rename(
            columns={
                "reviewerID": "userid",
                "asin": "movieid",
                "unixReviewTime": "timestamp",
            },
            inplace=True,
        )
        pcore_data.to_csv("./sasrec_emb/check.csv")
    else:
        pcore_data.rename(
            columns={"reviewerID": "userid", "asin": "itemid"}, inplace=True
        )
    return pcore_data


def get_movielens_data(
    local_file=None,
    download_path=None,
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
        zip_file_url = download_path
        with urllib.request.urlopen(zip_file_url) as zip_response:
            zip_contents = BytesIO(zip_response.read())
        print("downloaded")
    else:
        zip_contents = local_file
    ml_data = ml_genres = ml_tags = mapping = None
    print("done")
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
    # res[0].to_csv("./sasrec_emb/check_ml.csv")
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
    if not max_rank or max_rank > max(dataset_matrix.shape):
        max_rank = max(dataset_matrix.shape)
    if not min_rank or min_rank > max(dataset_matrix.shape):
        min_rank = min(dataset_matrix.shape)

    if not max_batch or max_batch > max(dataset_matrix.shape):
        max_batch = max(dataset_matrix.shape)
    if not min_batch or min_batch > max(dataset_matrix.shape):
        min_batch = min(dataset_matrix.shape)

    if var_name == "Rank":
        abs_min_rank = min_rank
        abs_max_rank = max_rank
        print(np.linspace(abs_min_rank, abs_max_rank, num_vals_rank, dtype=int))
        return np.linspace(abs_min_rank, abs_max_rank, num_vals_rank, dtype=int)

    if var_name == "Batch_size":
        if percents:
            abs_min_batch = dataset_matrix.shape[1] * min_batch // 100
            abs_max_batch = dataset_matrix.shape[1] * max_batch // 100
        else:
            abs_min_batch = min_batch
            abs_max_batch = max_batch

        return np.linspace(abs_min_batch, abs_max_batch, num_vals_batch_size, dtype=int)

    if var_name == "N_tries":
        return [25]
    if var_name == "Way":
        return ["top_rand", "CCL"]


def add_data(csv_path, dict_vals):
    """Adds data to csv file."""
    new_rows = []
    new_rows.append(dict_vals.values())
    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)


def time_func(func):
    def res_func(*args, **kwargs):
        time_start = timer()
        res = func(*args, **kwargs)
        time_finish = timer() - time_start
        print(time_finish)
        return res

    return res_func
