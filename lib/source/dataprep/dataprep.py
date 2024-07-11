from os import listdir
from os.path import isfile, join
from sklearn.utils.extmath import randomized_svd
from timeit import default_timer as timer
import numpy as np

import os

from lib.source.dataprep.utils import (
    get_movielens_data,
    matrix_from_observations,
    get_reviews,
)


def resolve_dataset_name(datafile, emb=False):
    """
    Resolve the base name of a dataset from its filename.

    Parameters:
    datafile (str): The filename of the dataset.
    emb (bool, optional): A flag indicating if the dataset is an embedding file. Defaults to False.

    Returns:
    str: The resolved dataset name without the file extension and additional suffixes.
    """
    dataset_name = ""
    if datafile[-2:] == "gz":
        dataset_name = datafile[:-10]
    elif datafile[-3:] == "zip":
        dataset_name = datafile[:-4]
    elif emb == True:
        dataset_name = datafile[:-13]
    return dataset_name


def dataset_preprocessing(dataset_name, datafile, datasets_dir):
    """
    Preprocess the dataset by loading and transforming it into a matrix of observations.

    Parameters:
    dataset_name (str): The name of the dataset.
    datafile (str): The filename of the dataset.
    datasets_dir (str): The directory where datasets are stored.

    Returns:
    numpy.ndarray: The matrix of observations derived from the dataset.
    """
    if dataset_name in ("ml-1m", "movieLens20m", "ml-20m"):
        full_path = os.path.join(datasets_dir, dataset_name + ".zip")
        print(full_path)
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


def svd_decomp(dataset_name, max_rank, matr_from_observ, svds, svd_dir):
    """
    Perform Singular Value Decomposition (SVD) on a matrix and handle caching of results.

    Parameters:
    dataset_name (str): The name of the dataset.
    max_rank (int): The maximum rank for the SVD.
    matr_from_observ (numpy.ndarray): The matrix of observations to decompose.
    svds (list): A list of existing SVD result filenames.
    svd_dir (str): The directory where SVD results are stored.

    Returns:
    tuple: A tuple containing the sorted singular values and the corresponding right singular vectors.
    """
    if (
        f"{dataset_name}_S_matrix_{max_rank}.npy"
        and f"{dataset_name}_V_matrix_{max_rank}.npy" in svds
    ):
        V_T = np.load(join(svd_dir, f"{dataset_name}_V_matrix_{max_rank}.npy"))
        S = np.load(join(svd_dir, f"{dataset_name}_S_matrix_{max_rank}.npy"))
    else:
        _, S, V_T = randomized_svd(matr_from_observ, n_components=max_rank)
        with open(
            join(svd_dir, f"{dataset_name}_S_matrix_{max_rank}.npy"), "wb+"
        ) as file:
            np.save(file, S)
        with open(
            join(svd_dir, f"{dataset_name}_V_matrix_{max_rank}.npy"), "wb+"
        ) as file:
            np.save(file, V_T)
    indices = np.flip(np.argsort(S))
    correct_S = [indices]  # randomized_svd not guarantees right order of eigen values
    correct_V_T = V_T[indices]
    return correct_S, correct_V_T
