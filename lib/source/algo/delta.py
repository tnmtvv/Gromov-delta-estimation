import numpy as np
import sys
import gc

from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix

from sklearn.metrics import pairwise_distances

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from utils import get_far_away_pairs, cuda_prep
import CCL
import condenced
import tensor
import true_delta

from numba import typed


def batched_delta_hyp(
    X,
    n_tries=10,
    batch_size=400,
    seed=42,
    economic=True,
    max_workers=25,
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
    rng = np.random.default_rng(seed)
    max_workers = max_workers or cpu_count() - 1
    t = 0
    mult_t = 0
    with ThreadPoolExecutor(max_workers=min(n_tries, max_workers)) as executor:
        futures = []
        for _ in range(n_tries):
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
    return results


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

    dist_matrix = pairwise_distances(X, metric="euclidean")
    del X
    gc.collect()
    print("matrix size: " + str(sys.getsizeof(dist_matrix)))

    diam = np.max(dist_matrix)

    if economic:
        if way in ["heuristic_CCL", "CCL", "GPU"]:
            far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 20)
        if way == "heuristic_CCL":
            print("pairs len " + str(len(far_away_pairs)))
            print("pairs size: " + str(sys.getsizeof(far_away_pairs)))
            delta = CCL.delta_CCL_heuristic(
                dist_matrix, typed.List(far_away_pairs), 100000
            )
        elif way == "CCL":
            print("CCL")
            delta = CCL.delta_hyp_condensed_CCL(typed.List(far_away_pairs), dist_matrix)
        elif way == "GPU":
            (
                n,
                x_coord_pairs,
                y_coord_pairs,
                adj_m,
                results,
                blockspergrid,
                threadsperblock,
            ) = cuda_prep(far_away_pairs, dist_matrix, 32)
            CCL.delta_hyp_CCL_GPU[blockspergrid, threadsperblock](
                n, x_coord_pairs, y_coord_pairs, adj_m, results
            )
        elif way == "rand_top":
            const = min(50, X.shape[0] - 1)
            delta = condenced.delta_hyp_condensed_rand_top(
                dist_matrix, X.shape[0], const, mode="top_rand"
            )
        elif way == "heuristic":
            const = min(50, X.shape[0] - 1)
            delta = condenced.delta_hyp_condensed_new(
                dist_matrix, X.shape[0], const, mode="top_rand"
            )
        elif way == "tensor":
            # used_indices = []
            objective_func = tensor.delta_protes(dist_matrix)
            delta = tensor.tensor_approximation(
                d=3, b_s=dist_matrix.shape[0], func=objective_func
            )
    else:
        delta = true_delta.delta_hyp(dist_matrix)
    delta_rel = 2 * delta / diam
    return delta_rel, diam
