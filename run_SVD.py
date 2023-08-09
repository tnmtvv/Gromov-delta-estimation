import argparse
import csv
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from sklearn.utils.extmath import randomized_svd
from utils import get_movielens_data, matrix_from_observations, get_reviews


def main(datasets_dir, svd_path, csv_path, rank):
    datafiles = [f for f in listdir(datasets_dir) if isfile(join(datasets_dir, f))]
    csv_file_name = "svd_time.csv"
    df = pd.DataFrame(
        columns=[
            "Dataset",
            "Time",
        ]
    )
    path_to_csv = os.path.join(csv_path, csv_file_name)
    df.to_csv(path_to_csv, index=False)
    for i, datafile in enumerate(datafiles):
        dataset_name = ""
        if datafile[-2:] == "gz":
            dataset_name = datafile[:-10]
        elif datafile[-3:] == "zip":
            dataset_name = datafile[:-10]
        print(dataset_name)
        if dataset_name == "Movielens_1m":
            cur_df = get_movielens_data(join(datasets_dir, datafile))
            matr_from_observ, u_id, i_id = matrix_from_observations(
                cur_df, dtype=float, itemid="movieid"
            )
        else:
            cur_df = get_reviews(join(datasets_dir, datafile))
            matr_from_observ, u_id, i_id = matrix_from_observations(cur_df, dtype=float)
        svd_time_start = timer()
        _, S, V = randomized_svd(matr_from_observ, n_components=rank)
        svd_time = timer() - svd_time_start

        with open(join(svd_path, f"{dataset_name}_S_matrix_{rank}.npy"), "wb+") as file:
            np.save(file, S)
        with open(join(svd_path, f"{dataset_name}_V_matrix_{rank}.npy"), "wb+") as file:
            np.save(file, V)
        new_rows = [
            [
                dataset_name,
                svd_time,
            ]
        ]
        with open(path_to_csv, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(new_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds csv files")
    parser.add_argument(
        "datasets_dir", type=str, help="Directory where datasets are stored"
    )
    parser.add_argument("svd_dir", type=str, help="Directory where svd are stored")
    parser.add_argument("csv_dir", type=str, help="Directory where csvs are stored")
    parser.add_argument("rank", type=int, help="rank to decompose matrix")

    args = parser.parse_args()
    main(args.datasets_dir, args.svd_dir, args.csv_dir, args.rank)
