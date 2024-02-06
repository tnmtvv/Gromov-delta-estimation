import gc
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from timeit import default_timer as timer

import numpy as np
from lib.source.algo.algo_utils import (
    calc_max_workers,
    cuda_prep,
    get_far_away_pairs,
    calc_max_lines,
    indx_to_2d,
    cuda_prep_cartesian,
    permutations,
    combinations,
)
from lib.source.algo.CCL import (
    delta_CCL_heuristic,
    delta_hyp_CCL_GPU,
    delta_hyp_condensed_CCL,
    delta_CCL_cartesian,
)
from lib.source.algo.condenced import delta_hyp_condensed, delta_hyp_condensed_heuristic
from lib.source.algo.tensor import delta_protes, tensor_approximation
from lib.source.algo.true_delta import delta_hyp
from line_profiler import profile
from numba import cuda, get_num_threads, njit, prange, typed
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances

# import libcontext


# from cupyx.scipy.spatial import distance_matrix
# import cupyx
# import cupy


# from memory_profiler import profile
# from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
# from cupyx.scipy.spatial import distance_matrix


def batched_delta_hyp(
    X,
    n_tries=10,
    batch_size=400,
    seed=42,
    economic=True,
    max_workers=25,
    mem_cpu_bound=16,
    mem_gpu_bound=16,
    way="heuristic",
):
    """
    Estimate the Gromov's delta hyperbolicity of a dataset using batch processing.

    Parameters
    ----------
    X : numpy.ndarray
        A 2D array of shape (n, m), where n is the number of nodes in a dataset and m is the dimensionality of the space
        that the nodes are embedded in.
    n_tries : int, optional
        The number of times to compute the delta hyperbolicity using different subsets of nodes.
    batch_size : int, optional
        The number of nodes to process in each batch.
    seed : int or None, optional
        Seed used for the random generator in batch sampling. Default is 42.
    economic : bool, optional
        If True, the function will use more memory-efficient methods. Default is True.
    max_workers : int or None, optional
        The maximum number of workers to use. If None, the number will be set to the number of available CPUs. Default is None.
    way : string
        Mode for calculations.

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples containing the delta hyperbolicity and diameter values of the dataset for different batches.

    Notes
    -----
    The function computes the delta hyperbolicity value of a dataset using batch processing.
    For each batch of nodes, the function computes the pairwise distances, computes the delta hyperbolicity value,
    and then aggregates the results across all batches to obtain the final delta hyperbolicity value.
    If economic=True, the function will use more efficient version of delta_hyp to combat better complexity.
    Pass way parameter to choose the mode.
    """
    print("true X shape" + str(X.shape))
    n_objects, _ = X.shape  # number of items
    results = []
    matrices_pairs = []
    x = cuda.device_array(1)  # resolving strange numba error
    rng = np.random.default_rng(seed)
    max_workers_cpu = min(
        max_workers, calc_max_workers(batch_size, mem_cpu_bound, n_tries)
    )
    if max_workers_cpu < 1:
        max_workers_cpu = 1
    max_workers_gpu = min(
        max_workers, calc_max_workers(batch_size, mem_gpu_bound, n_tries)
    )
    if max_workers_gpu < 25:
        way = "GPU_cartesian"

    if way != "GPU" and way != "GPU_cartesian":
        print("succsess")
        print("available num of theads " + str(max_workers_cpu))
        for part in range(int(n_tries // max_workers_cpu) + 1):
            if max_workers_cpu * part >= n_tries:
                break
            else:
                with ThreadPoolExecutor(
                    max_workers=min(n_tries - max_workers_cpu * part, max_workers_cpu)
                ) as executor:
                    futures = []
                    for _ in range(executor._max_workers):
                        if batch_size >= n_objects:
                            print("batch_size >= n_objects")
                            print(batch_size)
                            print(n_objects)
                            # `delta_hyp` selects a fixed point w.r.t which delta is computed
                            # the fixed point always corresponds to the first object, so shuffling allows
                            # exploring different initialization of a fixed point
                            batch_idx = rng.permutation(n_objects)
                        else:
                            batch_idx = rng.choice(
                                n_objects, batch_size, replace=False, shuffle=True
                            )
                        item_space = X[batch_idx]
                        print("batch done")
                        future = executor.submit(
                            delta_hyp_rel, item_space, economic=economic, way=way
                        )
                        futures.append(future)
                    for i, future in enumerate(as_completed(futures)):
                        delta_rel, diam = res = future.result()
                        print("res: " + str(res))
                        results.append(res)
    else:
        # in case of GPU way we precalculating all pairs and keep them on the CPU in order to not to overload the GPU memory
        print("available num of theads cpu " + str(max_workers_cpu))
        for part in range(int(n_tries // max_workers_cpu) + 1):
            if max_workers_cpu * part >= n_tries:
                break
            else:
                with ThreadPoolExecutor(
                    max_workers=min(n_tries - max_workers_cpu * part, max_workers_cpu)
                ) as executor:
                    futures = []
                    print(executor._max_workers)
                    for _ in range(executor._max_workers):
                        if batch_size >= n_objects:
                            print("batch_size >= n_objects")
                            print(batch_size)
                            print(n_objects)
                            # `delta_hyp` selects a fixed point w.r.t which delta is computed
                            # the fixed point always corresponds to the first object, so shuffling allows
                            # exploring different initialization of a fixed point
                            batch_idx = rng.permutation(n_objects)
                        else:
                            batch_idx = rng.choice(
                                n_objects, batch_size, replace=False, shuffle=True
                            )
                        item_space = X[batch_idx]
                        print("batch done")
                        future = executor.submit(preprocessing_for_GPU, item_space)
                        futures.append(future)
                    for _, future in enumerate(as_completed(futures)):
                        matrix, pairs = cur_matrix_and_pairs = future.result()
                        matrices_pairs.append(cur_matrix_and_pairs)
            print("matrices done")
            print("available num of theads gpu " + str(max_workers_gpu))
            if way == "GPU":
                for part_gpu in range(int(len(matrices_pairs) // max_workers_gpu)):
                    if max_workers_gpu * part_gpu >= len(matrices_pairs):
                        break
                    else:
                        with ThreadPoolExecutor(
                            max_workers=min(
                                n_tries - max_workers_gpu * part_gpu, max_workers_gpu
                            )
                        ) as executor_gpu:
                            futures = []
                            for i in range(executor_gpu._max_workers):
                                future = executor_gpu.submit(
                                    delta_hyp_GPU,
                                    matrices_pairs[part_gpu * max_workers_gpu + i][0],
                                    matrices_pairs[part_gpu * max_workers_gpu + i][1],
                                )
                                futures.append(future)
                            for j, future in enumerate(as_completed(futures)):
                                res = future.result()
                                print("res: " + str(res))
                                results.append(res)
            elif way == "GPU_cartesian":
                for i in range(len(matrices_pairs)):
                    max_lines = calc_max_lines(mem_gpu_bound, len(matrices_pairs[i][1]))
                    res = delta_cartesian_way(
                        matrices_pairs[i][0],
                        typed.List(matrices_pairs[i][1]),
                        max_lines,
                    )
                    results.append(res)
    return results


def preprocessing_for_GPU(X):
    # dist_matrix = squareform(cupy.ndarray.get(cupyx.scipy.spatial.distance.pdist(X)))
    dist_matrix = pairwise_distances(X, metric="euclidean")
    print(dist_matrix.shape)
    far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 20)
    return dist_matrix, far_away_pairs


def delta_hyp_GPU(dist_matrix, far_away_pairs):
    diam = np.max(dist_matrix)
    (
        n,
        x_coord_pairs,
        y_coord_pairs,
        adj_m,
        blockspergrid,
        threadsperblock,
        delta_res,
    ) = cuda_prep(far_away_pairs, dist_matrix, 32)
    print("pairs len " + str(len(far_away_pairs)))
    delta_hyp_CCL_GPU[blockspergrid, threadsperblock](
        n, x_coord_pairs, y_coord_pairs, adj_m, delta_res
    )
    del far_away_pairs
    del dist_matrix
    return 2 * delta_res[0] / diam, diam


# @njit(parallel=True)
@profile
def delta_cartesian_way(X, far_away_pairs, batch_size):
    diam = np.max(X)

    cartesian_size = int(len(far_away_pairs) * (len(far_away_pairs) - 1) / 2)

    batch_N = int(cartesian_size // batch_size) + 1
    deltas = np.zeros(batch_N)
    print(f"all_size: {cartesian_size}")
    print(f"batch_size: {batch_size}")

    for i in range(batch_N):
        print(f"{i} batch started")
        batch = prepare_batch(
            X,
            np.asarray(far_away_pairs),
            i * batch_size,
            min((i + 1) * batch_size, cartesian_size),
        )
        print("batch built")

        (
            cartesian_dist_array,
            delta_res,
            threadsperblock,
            blockspergrid,
        ) = cuda_prep_cartesian(batch, 32)
        delta_CCL_cartesian[blockspergrid, threadsperblock](
            cartesian_dist_array, delta_res
        )
        deltas[i] = delta_res[0]
        del cartesian_dist_array
    # print(deltas)
    delta = max(deltas)
    # print(delta)
    return 2 * delta / diam, diam


@profile
@njit(parallel=True)
def prepare_batch(X, far_away_pairs, start_ind, end_ind):
    batch = np.zeros((int(end_ind - start_ind), 6))
    for indx in prange(start_ind, end_ind):
        batch_line = np.zeros(6)
        i, j = indx_to_2d(indx)
        pair_1 = far_away_pairs[i]
        pair_2 = far_away_pairs[j]
        indices = [
            list(pair_1),
            list(pair_2),
            list([pair_1[0], pair_2[0]]),
            list([pair_1[1], pair_2[1]]),
            list([pair_1[0], pair_2[1]]),
            list([pair_1[1], pair_2[0]]),
        ]

        for k in prange(6):
            batch_line[k] = X[indices[k][0], indices[k][1]]

        batch[indx - start_ind] = batch_line

    return batch


# @profile
def delta_hyp_rel(X: np.ndarray, economic: bool = True, way="new"):
    """
    Computes relative delta hyperbolicity value and diameter from coordinates matrix.

    Parameters:
    -----------
    X : numpy.ndarray
        A 2D array of shape (n, m), where n is the number of nodes in a dataset and m is the dimensionality of the space
        that the nodes are embedded in.
    economic : bool, optional
        Whether to use the condensed distance matrix representation to compute the delta hyperbolicity value, by default True.
    way: string
        Which algo should be executed.

    Returns:
    --------
    Tuple[float, float]
        A tuple consisting of the relative delta hyperbolicity value (delta_rel) and the diameter of the manifold (diam).

    """
    if way == "condenced":
        dist_matrix = pdist(X)
    else:
        dist_matrix = pairwise_distances(X)
    diam = np.max(dist_matrix)

    if economic:
        if way in ["heuristic_CCL", "CCL", "GPU"]:
            far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 20)
        if way == "GPU":
            print("gpu")
            delta, _ = delta_hyp_GPU(dist_matrix, far_away_pairs)
            # delta = np.max(results)
        elif way == "heuristic_CCL":
            print("pairs len " + str(len(far_away_pairs)))
            print("pairs size: " + str(sys.getsizeof(far_away_pairs)))
            delta = delta_CCL_heuristic(dist_matrix, typed.List(far_away_pairs), 100000)
        elif way == "CCL":
            print("CCL")
            delta = delta_hyp_condensed_CCL(typed.List(far_away_pairs), dist_matrix)

        elif way == "rand_top":
            const = min(50, dist_matrix.shape[0] - 1)
            delta = delta_hyp_condensed_heuristic(
                dist_matrix, dist_matrix.shape[0], const, mode="top_rand"
            )
        elif way == "heuristic":
            const = min(50, dist_matrix.shape[0] - 1)
            delta = delta_hyp_condensed_heuristic(
                dist_matrix, dist_matrix.shape[0], const, mode="top_k"
            )
        elif way == "condenced":
            delta = delta_hyp_condensed(dist_matrix, X.shape[0])
        elif way == "tensor":
            # used_indices = []
            objective_func = delta_protes(dist_matrix)
            delta = tensor_approximation(
                d=3, b_s=dist_matrix.shape[0], func=objective_func
            )
    else:
        delta = delta_hyp(dist_matrix)
    delta_rel = 2 * delta / diam
    return delta_rel, diam


def deltas_comparison(
    X,
    n_tries=10,
    batch_size=400,
    seed=42,
    max_workers=25,
    way="heuristic",
):
    """
    Function for comparing delta, calculated with some heuristic method and ground truth delta (calculated with basic approach).

    Parameters
    ----------
    X : numpy.ndarray
      Item space matrix.
    n_tries : int, optional
        The number of times to compute the delta hyperbolicity using different subsets of nodes. Default is 10.
    batch_size : int, optional
        The number of nodes to process in each batch.
    seed : int or None, optional
        Seed used for the random generator in batch sampling. Default is 42.
    max_workers : int or None, optional
        The maximum number of workers to use. If None, the number will be set to the number of available CPUs. Default is None.
    way : string
        Mode for calculations.
    """
    rel_delta_start = timer()
    deltas_diams = batched_delta_hyp(
        X,
        n_tries=n_tries,
        batch_size=batch_size,
        seed=seed,
        economic=True,
        max_workers=max_workers,
        way=way,
    )
    rel_delta_time = timer() - rel_delta_start

    rel_deltas = list(map(lambda x: x[0], deltas_diams))
    rel_delta = np.mean(rel_deltas)

    true_delta_start = timer()
    true_delta = batched_delta_hyp(
        X,
        n_tries=n_tries,
        batch_size=batch_size,
        seed=seed,
        economic=False,
        max_workers=max_workers,
    )
    true_deltas = list(map(lambda x: x[0], true_delta))
    true_delta = np.mean(true_deltas)

    true_delta_time = timer() - true_delta_start

    print("---------------------------")
    print("true_delta " + str(true_delta))
    print("rel_delta " + str(rel_delta))

    print()
    print("true_delta time " + str(true_delta_time))
    print("rel_delta time " + str(rel_delta_time))

    print("---------------------------")
