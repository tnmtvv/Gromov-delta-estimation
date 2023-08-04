import argparse
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import yaml

from sklearn.utils.extmath import randomized_svd

from delta import batched_delta_hyp_old, batched_delta_hyp
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
    svd_dir,
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
    elif len(dependency) == 2:
        csv_file_name = f"{dependency[0]}_{dependency[1]}.csv"
    else:
        csv_file_name = f"{dependency[0]}.csv"

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
                svd_time_start = timer()
                if f"{dataset_name}_S_matrix_{max_rank}.npy" and f"{dataset_name}_V_matrix_{max_rank}.npy" in svds:
                    V = np.load(f"{dataset_name}_V_matrix.npy")
                    S = np.load(f"{dataset_name}_S_matrix.npy")
                else:
                    U, S, V = randomized_svd(matr_from_observ, n_components=max_rank)
                    with open(join(svd_dir, f"{dataset_name}_S_matrix_{max_rank}.npy"), "wb+") as file:
                        np.save(file, S)
                    with open(join(svd_dir, f"{dataset_name}_V_matrix_{max_rank}.npy"), "wb+") as file:
                        np.save(file, V)
                indices = np.flip(np.argsort(S))
                new_S = [S[i] for i in indices]
                svd_time = timer() - svd_time_start
                print("done svd, time: " + str(svd_time))

                for k, rank in enumerate(val_list_dict["Rank"]):
                    mult_time_start = timer()
                    item_space = V.T[:, indices[:rank]] @ np.diag(new_S[:rank])
                    mult_time = timer() - mult_time_start

                    for j, b_s in enumerate(val_list_dict["Batch_size"]):
                        for n_try in val_list_dict["N_tries"]:
                            old_time_start = timer()
                            deltas_diams = batched_delta_hyp_old(
                                item_space,
                                economic=True,
                                batch_size=b_s,
                                n_tries=n_try,
                                seed=42,
                            )
                            old_time = timer() - old_time_start
                            add_data(
                                deltas_diams=deltas_diams,
                                time=old_time + mult_time + svd_time,
                                k=rank,
                                mult_Time=mult_time,
                                svd_Time=svd_time,
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
                            new_time_start = timer()
                            deltas_diams = batched_delta_hyp(
                                matr_from_observ,
                                economic=True,
                                rank=rank,
                                batch_size=b_s,
                                n_tries=n_try,
                                seed=42,
                            )
                            new_time = timer() - new_time_start
                            add_data(
                                deltas_diams=deltas_diams,
                                time=new_time,
                                svd_Time=0,
                                mult_Time=0,
                                k=rank,
                                b_s=b_s,
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
                    print("done rank " + str(rank))
                print("done " + str(way))


def main(
    datasets_dir, csv_dir, svd_dir, path_to_config, min_batch, min_rank, max_batch, max_rank, grid_batch_size, grid_rank
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
            svd_dir,
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
    parser.add_argument("datasets_dir", type=str, help="Directory where datasets are stored")
    parser.add_argument("csv_dir", type=str, help="Directory where csvs are stored")
    parser.add_argument("svd_dir", type=str, help="Directory where matrices are stored")
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
        args.svd_dir,
        args.config_path,
        args.min_batch,
        args.min_rank,
        args.max_batch,
        args.max_rank,
        args.grid_batch_size,
        args.grid_rank
    )
