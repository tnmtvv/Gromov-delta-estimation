from os import listdir
from os.path import isfile, join
from sklearn.utils.extmath import randomized_svd
from timeit import default_timer as timer
import numpy as np

import os

from utils import (
    get_movielens_data,
    matrix_from_observations,
    get_reviews,
)


def resolve_dataset_name(datafile):
    dataset_name = ""
    if datafile[-2:] == "gz":
        dataset_name = datafile[:-10]
    elif datafile[-3:] == "zip":
        dataset_name = datafile[:-4]
    return dataset_name


def dataset_preprcessing(dataset_name, datafile, datasets_dir):
    if dataset_name in ("ml-1m", "movieLens20m"):
        full_path = os.path.join(datasets_dir, dataset_name)
        if os.path.exists(full_path):
            cur_df = get_movielens_data(full_path)
        else:
            download_path = (
                "http://files.grouplens.org/datasets/movielens/" + dataset_name
            )
            cur_df = get_movielens_data(download_path=download_path)
        matr_from_observ, _, _ = matrix_from_observations(
            cur_df, dtype=float, itemid="movieid"
        )
    else:
        cur_df = get_reviews(join(datasets_dir, datafile))
        matr_from_observ, _, _ = matrix_from_observations(cur_df, dtype=float)
    return matr_from_observ


def svd_decomp(dataset_name, max_rank, matr_from_observ, svds, svd_dir, verbose):
    if (
        f"{dataset_name}_S_matrix_{max_rank}.npy"
        and f"{dataset_name}_V_matrix_{max_rank}.npy" in svds
    ):  # if there are saved matrices, taking them from directory, else executing svd and save the result
        V = np.load(join(svd_dir, f"{dataset_name}_V_matrix_{max_rank}.npy"))
        S = np.load(join(svd_dir, f"{dataset_name}_S_matrix_{max_rank}.npy"))
    else:
        _, S, V = randomized_svd(matr_from_observ, n_components=max_rank)
        with open(
            join(svd_dir, f"{dataset_name}_S_matrix_{max_rank}.npy"), "wb+"
        ) as file:
            np.save(file, S)
        with open(
            join(svd_dir, f"{dataset_name}_V_matrix_{max_rank}.npy"), "wb+"
        ) as file:
            np.save(file, V)
    indices = np.flip(np.argsort(S))
    correct_S = [
        S[i] for i in indices
    ]  # randomized_svd not guarantees right order of eigen values
    return correct_S, V, indices
