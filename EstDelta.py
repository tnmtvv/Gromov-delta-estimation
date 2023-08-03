import argparse
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import yaml

from sklearn.utils.extmath import randomized_svd

from delta import batched_delta_hyp_old, batched_delta_hyp
import time
import os

from io import BytesIO

from utils import (
    get_movielens_data,
    matrix_from_observations,
    get_reviews,
    make_list_params,
    add_data,
)

try:
    import networkx as nx
except ImportError:
    nx = None
# import polara


def build_csv(
    datasets_dir,
    csv_dir,
    default_vals,
    dependency,
    min_batch=0,
    min_rank=0,
    max_batch=0,
    max_rank=0,
    num_vals_batch_size=10,
    num_vals_rank=10,
):
    if len(dependency) == 3:
        csv_file_name = (
            f"try_{dependency[0]}_{dependency[1]}_{dependency[2]}.csv"
        )
    else:
        csv_file_name = f"{dependency[0]}_{dependency[1]}.csv"
    datafiles = [f for f in listdir(datasets_dir) if isfile(join(datasets_dir, f))]
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
            "mult_Time",
            "Way",
        ]
    )
    path_to_csv = os.path.join(csv_dir, csv_file_name)
    df.to_csv(path_to_csv, index=False)

    datafiles.sort()

    val_list_dict = default_vals

    for i, datafile in enumerate(datafiles):
        dataset_name = ''
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
        abs_max_batch = max(matr_from_observ.shape)
        abs_max_rank = min(matr_from_observ.shape)

        for param in dependency:
            val_list_dict[param] = make_list_params(
                param,
                matr_from_observ,
                num_vals_batch_size,
                num_vals_rank,
                min_rank=min_rank,
                min_batch=min_batch,
                max_rank=max_rank,
                max_batch=max_batch,
            )

        for way in val_list_dict["Way"]:
            if way == "old":
                svd_time_start = time.time()
                U, S, V = randomized_svd(matr_from_observ, n_components=max_rank)
                indices = np.flip(np.argsort(S))
                new_S = [S[i] for i in indices]
                svd_time_end = time.time()
                print("done svd, time: " + str(svd_time_start - svd_time_end))

                for k, rank in enumerate(val_list_dict["Rank"]):
                    mult_time_start = time.time()
                    item_space = V.T[:, indices[:rank]] @ np.diag(new_S[:rank])
                    mult_time_end = time.time()

                    for j, b_s in enumerate(val_list_dict["Batch_size"]):
                        for n_try in val_list_dict["N_tries"]:
                            old_time_start = time.time()
                            deltas_diams = batched_delta_hyp_old(
                                item_space,
                                economic=True,
                                batch_size=b_s,
                                n_tries=n_try,
                                seed=42,
                            )
                            old_time_end = time.time()
                            add_data(
                                deltas_diams=deltas_diams,
                                time=(old_time_end - old_time_start) + (mult_time_end - mult_time_start) + (svd_time_end - svd_time_start),
                                k=max(rank, abs_max_rank),
                                mult_Time=(mult_time_end - mult_time_start),
                                svd_Time=(svd_time_end - svd_time_start),
                                b_s=b_s,
                                n_try=n_try,
                                way=way,
                                dataset=dataset_name,
                                csv_path=path_to_csv,
                            )
                            print("done try " + str(n_try))
                        print(
                            "done batch_size " +
                            str(b_s)
                        )
                    print("done rank " + str(rank))
                print("done " + str(way))
            if way == "new":
                for k, rank in enumerate(val_list_dict["Rank"]):
                    for j, b_s in enumerate(val_list_dict["Batch_size"]):
                        for n_try in val_list_dict["N_tries"]:
                            new_time_start = time.time()
                            deltas_diams = batched_delta_hyp(
                                matr_from_observ,
                                economic=True,
                                rank=rank,
                                batch_size=b_s,
                                n_tries=n_try,
                                seed=42,
                            )
                            new_time_end = time.time()
                            add_data(
                                deltas_diams=deltas_diams,
                                time=new_time_end - new_time_start,
                                svd_Time=0,
                                mult_Time=0,
                                k=max(rank, abs_max_rank),
                                b_s=max(b_s, abs_max_batch),
                                n_try=n_try,
                                way=way,
                                dataset=dataset_name,
                                csv_path=path_to_csv,
                            )
                            print("done try " + str(n_try))
                        print(
                            "done batch_size "
                            + str(b_s)
                        )
                    print("done rank " + str(max(rank, abs_max_rank)))
                print("done " + str(way))


def main(
    datasets_dir, csv_dir, path_to_config, min_batch, min_rank, max_batch, max_rank, grid_batch_size, grid_rank
):
    with open(path_to_config, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    dependencies = data["Dependencies"]
    default_values = data["Default_vals"]
    for dependency in dependencies:
        build_csv(
            datasets_dir,
            csv_dir,
            default_values,
            dependency,
            min_batch,
            min_rank,
            max_batch,
            max_rank,
            num_vals_batch_size=grid_batch_size,
            num_vals_rank=grid_rank
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds csv files")
    parser.add_argument(
        "datasets_dir", type=str, help="Directory where datasets are stored"
    )
    parser.add_argument("csv_dir", type=str, help="Directory where csvs are stored")
    parser.add_argument("config_path", type=str, help="Path ro config file")
    parser.add_argument("min_batch", type=int, help="Minimum batch size")
    parser.add_argument("min_rank", type=int, help="Minimum rank")
    parser.add_argument("max_batch", type=int, help="Maximum batch size")
    parser.add_argument("max_rank", type=int, help="Maximum rank")
    parser.add_argument("grid_batch_size", type=int, help="Num estimations on batch")
    parser.add_argument("grid_rank", type=int, help="Num estimations on rank")

    args = parser.parse_args()
    main(
        args.datasets_dir,
        args.csv_dir,
        args.config_path,
        args.min_batch,
        args.min_rank,
        args.max_batch,
        args.max_rank,
        args.grid_batch_size,
        args.grid_rank
    )
