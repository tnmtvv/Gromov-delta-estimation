import argparse
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import yaml

from sklearn.utils.extmath import randomized_svd

from delta import batched_delta_hyp, deltas_comparison
import os


from utils import (
    get_movielens_data,
    matrix_from_observations,
    get_reviews,
    make_list_params,
    add_data,
)


def build_csv(
    datasets_dir,
    csv_dir,
    csv_file_name,
    svd_dir,
    default_vals,
    dependency,
    min_batch=0,
    min_rank=0,
    max_batch=0,
    max_rank=0,
    num_vals_batch_size=10,
    num_vals_rank=10,
    compare=False,
    verbose=True,
    percents=True,
):
    """Execute all the experiments according dependencies written in the delta_config file and fills csv file."""
    datafiles = [f for f in listdir(datasets_dir) if isfile(join(datasets_dir, f))]
    svds = [f for f in listdir(svd_dir) if isfile(join(svd_dir, f))]

    rng = np.random.default_rng(42)

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

    for i, datafile in enumerate(datafiles):
        dataset_name = ""
        if datafile[-2:] == "gz":
            dataset_name = datafile[:-10]
        elif datafile[-3:] == "zip":
            dataset_name = datafile[:-4]
        if verbose:
            print(dataset_name)
        if dataset_name in ("ml-1m", "movieLens20m"):
            cur_df = get_movielens_data("datasets/Zip/movieLens20m.zip")
            # cur_df = get_movielens_data("C:\work\GitHub\DeltaEstimation\datasets\movieLens20m.zip")
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
                percents=percents,
            )
        print(val_list_dict["Batch_size"])
        print(val_list_dict["N_tries"])
        print(val_list_dict["Rank"])
        max_rank = np.max(val_list_dict["Rank"])
        for way in val_list_dict["Way"]:
            # if way == "old":
            svd_time_start = timer()
            if (
                f"{dataset_name}_S_matrix_{max_rank}.npy"
                and f"{dataset_name}_V_matrix_{max_rank}.npy" in svds
            ):
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
            new_S = [S[i] for i in indices]
            svd_time = timer() - svd_time_start
            if verbose:
                print("done svd, time: " + str(svd_time))
            else:
                svd_time = 0
            for k, rank in enumerate(val_list_dict["Rank"]):
                # if way == "old":
                item_space = V.T[:, indices[:rank]] @ np.diag(new_S[:rank])
                random_users = rng.choice(
                    matr_from_observ.shape[0], rank, replace=False, shuffle=True
                )
                # item_space = matr_from_observ.toarray()
                # print("itemspace " + str(item_space.size))
                # else:
                # item_space = matr_from_observ
                for j, b_s in enumerate(val_list_dict["Batch_size"]):
                    for n_try in val_list_dict["N_tries"]:
                        if not compare:
                            delta_time_start = timer()
                            print("delta start")
                            deltas_diams = batched_delta_hyp(
                                item_space,
                                economic=True,
                                max_workers=50,
                                batch_size=b_s,
                                n_tries=n_try,
                                seed=42,
                                way=way,
                            )

                            delta_time = timer() - delta_time_start

                            deltas = list(map(lambda x: x[0], deltas_diams))
                            diams = list(map(lambda x: x[1], deltas_diams))

                            print("cur_mean " + str(np.mean(deltas)))
                            for d_idx, delta in enumerate(deltas):
                                add_data(
                                    path_to_csv,
                                    {
                                        "Delta": delta,
                                        "Diam": diams[d_idx],
                                        "Dataset": dataset_name,
                                        "Mean_delta": np.mean(deltas),
                                        "Std_delta": np.asarray(deltas).std(ddof=1),
                                        "Rank": rank,
                                        "Batch_size": b_s,
                                        "Num_of_attempts": n_try,
                                        "Mean_diam": np.mean(diams),
                                        "Std_diam": np.std(diams),
                                        "all_Time": svd_time + delta_time,
                                        "svd_Time": svd_time,
                                        "Way": way,
                                    },
                                )
                        else:
                            deltas_comparison(
                                item_space,
                                n_tries=10,
                                batch_size=400,
                                seed=42,
                                max_workers=25,
                                rank=10,
                            )
                        if verbose:
                            print("done try " + str(n_try))
                    if verbose:
                        print("done batch_size " + str(b_s))
                if verbose:
                    print("done rank " + str(rank))
            if verbose:
                print("done " + str(way))


def main(
    csv_name,
    path_to_config,
    min_batch,
    min_rank,
    max_batch,
    max_rank,
    grid_batch_size,
    grid_rank,
    compare,
    verbose,
    percents,
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
        build_csv(
            datasets_dir,
            csv_dir,
            csv_name,
            svd_dir,
            default_values,
            dependency,
            min_batch,
            min_rank,
            max_batch,
            max_rank,
            num_vals_batch_size=grid_batch_size,
            num_vals_rank=grid_rank,
            compare=compare,
            verbose=verbose,
            percents=percents,
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
        args.min_batch,
        args.min_rank,
        args.max_batch,
        args.max_rank,
        args.grid_batch_size,
        args.grid_rank,
        args.compare,
        args.verbose,
        args.percents,
    )
