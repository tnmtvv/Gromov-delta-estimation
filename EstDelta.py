import argparse
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer
from scipy.spatial.distance import pdist
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from sklearn.utils.extmath import randomized_svd

from delta import batched_delta_hyp, deltas_comparison
import os


from utils import (
    get_movielens_data,
    matrix_from_observations,
    get_reviews,
)


def build_hist(datafile, datasets_dir, svd_dir, svds, rank=3000, verbose=True):
    dataset_name = ""
    if datafile[-2:] == "gz":
        dataset_name = datafile[:-10]
    elif datafile[-3:] == "zip":
        dataset_name = datafile[:-4]
    if verbose:
        print(dataset_name)
    if dataset_name in ("ml-1m", "movieLens20m"):
        cur_df = get_movielens_data()
        # cur_df = get_movielens_data("C:\work\GitHub\DeltaEstimation\datasets\movieLens20m.zip")
        matr_from_observ, u_id, i_id = matrix_from_observations(
            cur_df, dtype=float, itemid="movieid"
        )
    else:
        cur_df = get_reviews(join(datasets_dir, datafile))
        matr_from_observ, u_id, i_id = matrix_from_observations(cur_df, dtype=float)
        # if way == "old":
    if (
        f"{dataset_name}_S_matrix_{rank}.npy"
        and f"{dataset_name}_V_matrix_{rank}.npy" in svds
    ):
        V = np.load(join(svd_dir, f"{dataset_name}_V_matrix_{rank}.npy"))
        S = np.load(join(svd_dir, f"{dataset_name}_S_matrix_{rank}.npy"))
    else:
        _, S, V = randomized_svd(matr_from_observ, n_components=rank)
        with open(join(svd_dir, f"{dataset_name}_S_matrix_{rank}.npy"), "wb+") as file:
            np.save(file, S)
        with open(join(svd_dir, f"{dataset_name}_V_matrix_{rank}.npy"), "wb+") as file:
            np.save(file, V)
    indices = np.flip(np.argsort(S))
    new_S = [S[i] for i in indices]
    item_space = V.T[:, indices[:3000]] @ np.diag(new_S[:3000])
    print("start")
    dist_condensed = pdist(item_space)
    print("end")

    fig, ax = plt.subplots(figsize=(10, 10))
    n_bins = 1000
    # plt.yscale("log")
    ax.hist(
        dist_condensed.flatten(),
        bins=n_bins,
        color="lightblue",
    )
    ax.set_title("dists")
    plt.savefig(str(n_bins) + "_saved_hist_" + dataset_name + ".png")


def build_hist(
    datasets_dir,
    csv_dir,
    csv_file_name,
    svd_dir,
    default_vals,
):
    """Execute all the experiments according dependencies written in the delta_config file and fills csv file."""
    datafiles = [f for f in listdir(datasets_dir) if isfile(join(datasets_dir, f))]
    svds = [f for f in listdir(svd_dir) if isfile(join(svd_dir, f))]

    df = pd.DataFrame(
        columns=[
            "Delta",
            "Diam",
            "Dataset",
            "Mean_delta",
            "Std_delta",
            "Rank",
            "Batch_size",
            "Num_of_attempts",
            "Mean_diam",
            "Std_diam",
            "all_Time",
            "svd_Time",
            "Way",
        ]
    )
    path_to_csv = os.path.join(csv_dir, csv_file_name)
    df.to_csv(path_to_csv, index=False)

    val_list_dict = default_vals

    with ThreadPoolExecutor(max_workers=len(datafiles)) as executor:
        futures = []
        for i in range(len(datafiles)):
            future = executor.submit(
                build_hist,
                datafiles[i],
                datasets_dir,
                svd_dir,
                svds,
            )
            futures.append(future)
            for i, future in enumerate(as_completed(futures)):
                # delta_rel, diam = res = future.result()
                future.result()
def main(
    csv_name,
    path_to_config,
):
    with open(path_to_config, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    dependencies = data["Dependencies"]
    default_values = data["Default_vals"]
    datasets_dir = data["Path_to_datasets"]
    csv_dir = data["Path_to_csvs"]
    svd_dir = data["Path_to_svds"]

    for dependency in dependencies:
        build_hist(
            datasets_dir,
            csv_dir,
            csv_name,
            svd_dir,
            default_values,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds csv files")
    parser.add_argument("csv_name", type=str, help="Name of the future csv file")
    parser.add_argument("config_path", type=str, help="Path ro config file")
    parser.add_argument("min_batch", type=int, help="Minimum batch size")
    parser.add_argument("min_rank", type=int, help="Minimum rank")
    parser.add_argument("max_batch", type=int, help="Maximum batch size")
    parser.add_argument("max_rank", type=int, help="Maximum rank")
    parser.add_argument("grid_batch_size", type=int, help="Num estimations on batch")
    parser.add_argument("grid_rank", type=int, help="Num estimations on rank")
    parser.add_argument(
        "-p",
        dest="percents",
        action="store_true",
        help="Batch_size boudaries provided as percents or as absolute values",
    )
    parser.add_argument(
        "-v", dest="verbose", action="store_true", help="Flag to print log"
    )
    parser.add_argument(
        "-c", dest="compare", action="store_true", help="Flag to compare results"
    )
    args = parser.parse_args()
    main(
        args.csv_name,
        args.config_path,
    )
