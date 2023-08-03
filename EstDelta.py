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
    max_batch=0,
    max_rank=0,
    num_vals=10,
    up_bound=10,
    first_dependency="Rank",
    second_dependency="Batch_size",
    third_dependency=None,
):
    if third_dependency:
        csv_file_name = (
            f"try_{first_dependency}_{second_dependency}_{third_dependency}.csv"
        )
    else:
        csv_file_name = f"{first_dependency}_{second_dependency}.csv"
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
        if datafile[-2:] == "gz":
            print(datafile[:-10])
        elif datafile[-3:] == "zip":
            print(datafile[:-4])
        if datafile[:-4] == "Movielens_1m":
            cur_df = get_movielens_data(join(datasets_dir, datafile))
            matr_from_observ, u_id, i_id = matrix_from_observations(
                cur_df, dtype=float, itemid="movieid"
            )
        else:
            cur_df = get_reviews(join(datasets_dir, datafile))
            matr_from_observ, u_id, i_id = matrix_from_observations(cur_df, dtype=float)
        abs_max_batch = max(matr_from_observ.shape)
        abs_max_rank = min(matr_from_observ.shape)

        val_list_dict[first_dependency] = make_list_params(
            first_dependency,
            matr_from_observ,
            num_vals,
            up_bound,
            max_rank=max_rank,
            max_batch=max_batch,
        )
        val_list_dict[second_dependency] = make_list_params(
            second_dependency,
            matr_from_observ,
            num_vals,
            up_bound,
            max_rank=max_rank,
            max_batch=max_batch,
        )
        if third_dependency:
            val_list_dict[third_dependency] = make_list_params(
                third_dependency,
                matr_from_observ,
                num_vals,
                up_bound,
                max_rank=max_rank,
                max_batch=max_batch,
            )
        for way in val_list_dict["Way"]:
            if way == "old":
                st_1 = time.time()
                U, S, V = randomized_svd(matr_from_observ, n_components=max_rank)
                indices = np.flip(np.argsort(S))
                new_S = [S[i] for i in indices]
                et_1 = time.time()
                print("done svd, time: " + str(et_1 - st_1))

                for k, rank in enumerate(val_list_dict["Rank"]):
                    st_2 = time.time()
                    item_space = V.T[:, indices[:rank]] @ np.diag(new_S[:rank])
                    et_2 = time.time()

                    for j, b_s in enumerate(val_list_dict["Batch_size"]):
                        for n_try in val_list_dict["N_tries"]:
                            st_old = time.time()
                            deltas_diams = batched_delta_hyp_old(
                                item_space,
                                economic=True,
                                batch_size=b_s,
                                n_tries=n_try,
                                seed=42,
                            )
                            et_old = time.time()
                            add_data(
                                deltas_diams=deltas_diams,
                                time=(et_old - st_old) + (et_1 - st_1) + (et_2 - st_2),
                                k=rank,
                                mult_Time=(et_2 - st_2),
                                svd_Time=(et_1 - st_1),
                                b_s=(b_s * 100) // abs_max_batch,
                                n_try=n_try,
                                way=way,
                                dataset=datafile[:-10],
                                csv_path=path_to_csv,
                            )
                            print("done try " + str(n_try))
                        print(
                            "done batch_size "
                            + str((b_s * 100) // abs_max_batch)
                            + " %"
                        )
                    print("done rank " + str(rank))
                print("done " + str(way))
            if way == "new":
                for k, rank in enumerate(val_list_dict["Rank"]):
                    for j, b_s in enumerate(val_list_dict["Batch_size"]):
                        for n_try in val_list_dict["N_tries"]:
                            st = time.time()
                            deltas_diams = batched_delta_hyp(
                                matr_from_observ,
                                economic=True,
                                rank=rank,
                                batch_size=b_s,
                                n_tries=n_try,
                                seed=42,
                            )
                            et = time.time()
                            add_data(
                                deltas_diams=deltas_diams,
                                time=et - st,
                                svd_Time=0,
                                mult_Time=0,
                                k=rank,
                                b_s=(b_s * 100) // abs_max_batch,
                                n_try=n_try,
                                way=way,
                                dataset=datafile[:-10],
                                csv_path=path_to_csv,
                            )
                            print("done try " + str(n_try))
                        print(
                            "done batch_size "
                            + str((b_s * 100) // abs_max_batch)
                            + " %"
                        )
                    print("done rank " + str(rank))
                print("done " + str(way))


def main(
    datasets_dir, csv_dir, path_to_config, max_batch, max_rank, grid_size, up_bound
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
            max_batch,
            max_rank,
            num_vals=grid_size,
            up_bound=up_bound,
            first_dependency=dependency[0],
            second_dependency=dependency[1],
            third_dependency="Way",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds csv files")
    parser.add_argument(
        "datasets_dir", type=str, help="Directory where datasets are stored"
    )
    parser.add_argument("csv_dir", type=str, help="Directory where csvs are stored")
    parser.add_argument("config_path", type=str, help="Path ro config file")
    parser.add_argument("max_batch", type=int, help="Maximum batch size")
    parser.add_argument("max_rank", type=int, help="Maximum rank")
    parser.add_argument("grid_size", type=int, help="num points on each axis")
    parser.add_argument("up_bound", type=int, help="num factual points on each axis")

    args = parser.parse_args()
    main(
        args.datasets_dir,
        args.csv_dir,
        args.config_path,
        args.max_batch,
        args.max_rank,
        args.grid_size,
        args.up_bound,
    )
