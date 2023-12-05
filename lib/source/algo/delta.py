import numpy as np
import sys
import gc

# import libcontext

from numba import typed, get_num_threads, cuda, njit, prange
from timeit import default_timer as timer
from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import pdist
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# from line_profiler import profile

from lib.source.algo.algo_utils import get_far_away_pairs, cuda_prep, calc_max_workers
from lib.source.algo.CCL import (
    delta_CCL_heuristic,
    delta_hyp_condensed_CCL,
    delta_hyp_CCL_GPU,
)
from lib.source.algo.condenced import (
    delta_hyp_condensed_heuristic,
    delta_hyp_condensed,
)
from lib.source.algo.tensor import delta_protes, tensor_approximation

from lib.source.algo.true_delta import delta_hyp

# from memory_profiler import profile
# from dist_matrix.cuda_dist_matrix_full import dist_matrix as gpu_dist_matrix
# from cupyx.scipy.spatial import distance_matrix


# @profile
def batched_delta_hyp(
    X,
    n_tries=10,
    batch_size=400,
    seed=42,
    economic=True,
    max_workers=25,
    mem_bound=100,
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
    x = cuda.device_array(1)  # resolving strange numba error
    rng = np.random.default_rng(seed)
    max_workers = min(max_workers, calc_max_workers(batch_size, mem_bound, n_tries))
    print("succsess")
    print("available num of theads " + str(max_workers))
    for part in range(int(n_tries // max_workers) + 1):
        if max_workers * part >= n_tries:
            break
        else:
            with ThreadPoolExecutor(
                max_workers=min(n_tries - max_workers * part, max_workers)
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
    # else:
    #     far_away_pairs = []
    #     dist_matrices = []
    #     for _ in range(n_tries):
    #         dist_matrix = pairwise_distances(X, metric="euclidean")
    #         dist_matrices.append(dist_matrix)
    #         far_away_pairs.append(
    #             get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 20)
    #         )
    #     results = delta_hyp_GPU(n_tries, far_away_pairs, dist_matrices)
    return results


def delta_hyp_GPU(
    far_away_pairs,
    dist_matrix,
):
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
    return delta_res[0]


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

    if way != "condenced":
        dist_matrix = pairwise_distances(X, metric="euclidean")
    else:
        dist_matrix = pdist(X)
    diam = np.max(dist_matrix)

    if economic:
        if way in ["heuristic_CCL", "CCL", "GPU"]:
            far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 20)
        if way == "heuristic_CCL":
            print("pairs len " + str(len(far_away_pairs)))
            print("pairs size: " + str(sys.getsizeof(far_away_pairs)))
            delta = delta_CCL_heuristic(dist_matrix, typed.List(far_away_pairs), 100000)
        elif way == "CCL":
            print("CCL")
            delta = delta_hyp_condensed_CCL(typed.List(far_away_pairs), dist_matrix)
        elif way == "GPU":
            delta = delta_hyp_GPU(far_away_pairs, dist_matrix)
            # delta = np.max(results)
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
