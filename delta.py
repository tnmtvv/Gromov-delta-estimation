import numpy as np
import time

from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from scipy.spatial.distance import pdist, squareform
from numba import njit, prange, set_num_threads

try:
    import networkx as nx
except ImportError:
    nx = None


def delta_hyp(dismat: np.ndarray) -> float:
    """
    Computes Gromov's delta-hyperbolicity value from distance matrix using the maxmin product.

    Parameters:
    -----------
    dismat : numpy.ndarray
        A square distance matrix of shape (n, n), where n is the number of nodes in a network.

    Returns:
    --------
    float
        The delta hyperbolicity value.

    Notes:
    ------
    This is a naive implementation that can be very inefficient on large datasets.
    Use `delta_hyp_condensed` function for scalable computations.
    """

    p = 0  # fixed point (hence the dataset should be shuffled for more reliable results)
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)

    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)


# def sample_hyperbolicity(adj_m, num_samples=100, seed=42):
#     hyps = []
#     for i in range(num_samples):
#         rng = np.random.default_rng()
#         node_tuple = rng.integers(low=0, high=adj_m.shape[0] - 1, size=4)
#         try:
#             d01 = adj_m[node_tuple[0], node_tuple[1]]
#             d23 = adj_m[node_tuple[2], node_tuple[3]]
#             d02 = adj_m[node_tuple[0], node_tuple[2]]
#             d13 = adj_m[node_tuple[1], node_tuple[3]]
#             d03 = adj_m[node_tuple[0], node_tuple[3]]
#             d12 = adj_m[node_tuple[1], node_tuple[2]]
#
#             s = [d01 + d23, d02 + d13, d03 + d12]
#             s.sort()
#             hyps.append((s[-1] - s[-2]))
#         except Exception as e:
#             continue
#
#     return 0.5 * np.max(hyps), 0.5 * np.mean(hyps)


# @profile
def batched_delta_hyp(
    X,
    n_tries=10,
    batch_size=400,
    seed=None,
    economic=True,
    max_workers=5,
    rank=10,
    way="old",
):
    """
    Estimate the Gromov's delta hyperbolicity of a network using batch processing.

    Parameters
    ----------
    X : numpy.ndarray
        A 2D array of shape (n, m), where n is the number of nodes in a network and m is the dimensionality of the space
        that the nodes are embedded in.
    n_tries : int, optional
        The number of times to compute the delta hyperbolicity using different subsets of nodes. Default is 10.
    batch_size : int, optional
        The number of nodes to process in each batch. Default is 1500.
    seed : int or None, optional
        Seed used for the random generator in batch sampling. Default is None.
    economic : bool, optional
        If True, the function will use more memory-efficient methods. Default is True.
    max_workers : int or None, optional
        The maximum number of workers to use. If None, the number will be set to the number of available CPUs. Default is None.

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples containing the delta hyperbolicity and diameter values of the network for different batches.

    Notes
    -----
    The function computes the delta hyperbolicity value of a network using batch processing.
    For each batch of nodes, the function computes the pairwise distances, computes the delta hyperbolicity value,
    and then aggregates the results across all batches to obtain the final delta hyperbolicity value.
    If economic=True, the function will use more efficient version of delta_hyp to combat O(n^3) complexity.
    """
    if way == "old":
        n_objects, _ = X.shape
    else:
        _, n_objects = X.shape
    results = []
    # logger.info(f'Calculating delta on {min(batch_size, n_objects)} samples out of {n_objects}.')
    rng = np.random.default_rng(seed)
    max_workers = max_workers or cpu_count() - 1
    t = 0
    mult_t = 0
    with ThreadPoolExecutor(max_workers=min(n_tries, max_workers)) as executor:
        futures = []
        for _ in range(n_tries):
            if batch_size >= n_objects:
                # `delta_hyp` selects a fixed point w.r.t which delta is computed
                # the fixed point always corresponds to the first object, so shuffling allows
                # exploring different initialization of a fixed point
                batch_idx = rng.permutation(n_objects)
                if way == "old":
                    batched_matr = csr_matrix(X[batch_idx])
                else:
                    batched_matr = csr_matrix(X[:, batch_idx])
            else:
                batch_idx = rng.choice(
                    n_objects, batch_size, replace=False, shuffle=True
                )
                if way == "old":
                    batched_matr = csr_matrix(X[batch_idx])
                else:
                    batched_matr = csr_matrix(X[:, batch_idx])
            if way == "old":
                item_space = X[batch_idx]
            else:
                U, S, V = randomized_svd(batched_matr, n_components=rank)
                indices = np.flip(np.argsort(S))
                new_S = [S[i] for i in indices]
                item_space = V.T[:, indices[:rank]] @ np.diag(new_S)
            future = executor.submit(delta_hyp_rel, item_space, economic=economic)
            futures.append(future)
        for i, future in enumerate(as_completed(futures)):
            delta_rel, diam = res = future.result()
            # logger.info(f'Trial {i + 1}/{n_tries} relative delta: {delta_rel} for estimated diameter: {diam}')
            results.append(res)
    return results


#@profile
def delta_hyp_rel(X: np.ndarray, economic: bool = True):
    """
    Computes relative delta hyperbolicity value and diameter from coordinates matrix.

    Parameters:
    -----------
    X : numpy.ndarray
        A 2D array of shape (n, m), where n is the number of nodes in a network and m is the dimensionality of the space
        that the nodes are embedded in.
    economic : bool, optional
        Whether to use the condensed distance matrix representation to compute the delta hyperbolicity value, by default True.

    Returns:
    --------
    Tuple[float, float]
        A tuple consisting of the relative delta hyperbolicity value (delta_rel) and the diameter of the manifold (diam).

    """
    dist_condensed = pdist(X)
    # dist_condensed = pairwise_distances(X)
    diam = np.max(dist_condensed)
    if economic:
        # sq_matrix = squareform(dist_condensed)
        delta = delta_hyp_condensed_new(squareform(dist_condensed), X.shape[0])
        # print('tries ' + str(tries))
        # delta, _ = sample_hyperbolicity(squareform(dist_condensed))
    else:
        delta = delta_hyp(squareform(dist_condensed))
    delta_rel = 2 * delta / diam
    return delta_rel, diam


@njit(parallel=True)
def delta_hyp_condensed(dist_condensed: np.ndarray, n_samples: int) -> float:
    """
    Compute the delta hyperbolicity value from the condensed distance matrix representation.
    This is a more efficient analog of the `delta_hyp` function.

    Parameters
    ----------
    dist_condensed : numpy.ndarray
        A 1D array representing the condensed distance matrix of the network.
    n_samples : int
        The number of nodes in the network.

    Returns
    -------
    float
        The delta hyperbolicity of the network.

    Notes
    -----
    Calculation heavily relies on the `scipy`'s `pdist` output format. According to the docs (as of v.1.10.1):
    "The metric dist(u=X[i], v=X[j]) is computed and stored in entry m * i + j - ((i + 2) * (i + 1)) // 2."
    Additionally, it implicitly assumes that j > i. Note that dist(u=X[0], v=X[k]) is defined by (k - 1)'s entry.
    """
    delta_hyp = np.zeros(n_samples, dtype=dist_condensed.dtype)
    for k in prange(n_samples):
        # as in `delta_hyp`, fixed point is selected at 0
        delta_hyp_k = 0.0
        dist_0k = 0.0 if k == 0 else dist_condensed[k - 1]
        for i in range(n_samples):
            if i == 0:
                dist_0i = 0.0
                dist_ik = dist_0k
            else:
                if k == i:
                    dist_0i = dist_0k
                    dist_ik = 0.0
                else:
                    dist_0i = dist_condensed[i - 1]
                    i1, i2 = (i, k) if k > i else (k, i)
                    ik_idx = n_samples * i1 + i2 - ((i1 + 2) * (i1 + 1)) // 2
                    dist_ik = dist_condensed[int(ik_idx)]
            diff_ik = dist_0i - dist_ik
            for j in range(i, n_samples):
                if j == 0:
                    dist_0j = 0.0
                    dist_jk = dist_0k
                else:
                    if k == j:
                        dist_0j = dist_0k
                        dist_jk = 0.0
                    else:
                        dist_0j = dist_condensed[j - 1]
                        j1, j2 = (j, k) if k > j else (k, j)
                        jk_idx = n_samples * j1 + j2 - ((j1 + 2) * (j1 + 1)) // 2
                        dist_jk = dist_condensed[int(jk_idx)]
                diff_jk = dist_0j - dist_jk
                if i == j:
                    dist_ij = 0.0
                else:
                    ij_idx = (
                        n_samples * i + j - ((i + 2) * (i + 1)) // 2
                    )  # j >= i by construction
                    dist_ij = dist_condensed[int(ij_idx)]
                gromov_ij = dist_0i + dist_0j - dist_ij
                delta_hyp_k = max(
                    delta_hyp_k, dist_0k + min(diff_ik, diff_jk) - gromov_ij
                )
        delta_hyp[k] = delta_hyp_k
    return 0.5 * np.max(delta_hyp)


@njit(parallel=True)
def delta_hyp_condensed_new(dist_condensed: np.ndarray, n_samples: int) -> float:
    """
    Compute the delta hyperbolicity value from the condensed distance matrix representation.
    This is a more efficient analog of the `delta_hyp` function.

    Parameters
    ----------
    dist_condensed : numpy.ndarray
        A 1D array representing the condensed distance matrix of the network.
    n_samples : int
        The number of nodes in the network.

    Returns
    -------
    float
        The delta hyperbolicity of the network.

    Notes
    -----
    Calculation heavily relies on the `scipy`'s `pdist` output format. According to the docs (as of v.1.10.1):
    "The metric dist(u=X[i], v=X[j]) is computed and stored in entry m * i + j - ((i + 2) * (i + 1)) // 2."
    Additionally, it implicitly assumes that j > i. Note that dist(u=X[0], v=X[k]) is defined by (k - 1)'s entry.
    """
    delta_hyp = np.zeros(n_samples, dtype=dist_condensed.dtype)
    for k in prange(n_samples):
        # as in `delta_hyp`, fixed point is selected at 0
        delta_hyp_k = 0.0
        dist_0k = dist_condensed[0][k - 1]

        # сортим расстояния от k до остальных точек, индекс i будем выбирать из самых дальних
        inds_i = np.argsort(dist_condensed[k - 1])

        # брать 10 процентов от размера выборки ?
        for ind_i in inds_i[-15:]:
            dist_0i = dist_condensed[0][ind_i]
            dist_ik = dist_condensed[ind_i][k - 1]

            # сортим расстояния от i до остальных точек, индекс j будем выбирать из самых дальних
            inds_j = np.argsort(dist_condensed[:, ind_i])

            for ind_j in inds_j[-15:]:
                dist_0j = dist_condensed[0][ind_j]
                dist_jk = dist_condensed[ind_j][k - 1]
                dist_ij = dist_condensed[ind_i][ind_j]

                # алгоритм с S
                dist_array = [dist_0j + dist_ik, dist_0i + dist_jk, dist_0k + dist_ij]
                s1 = max(dist_array)
                dist_array.remove(s1)
                s2 = max(dist_array)
                delta_hyp_k = max(
                    delta_hyp_k, s1 - s2
                )
        delta_hyp[k] = delta_hyp_k
    return 0.5 * np.max(delta_hyp)


def relative_delta_poincare(tol=1e-5):
    """
    Computes relative delta-hyperbolicity for a Poincar'e disk within the given machine precision.

    Notes:
    -----
    $$
    \delta_{\text{rel}} = \frac{2\,\delta_P}{\operatorname{diam}(P)}

    \delta_P = \ln(1+\sqrt{2}),\quad \operatorname{diam}(P) = 2 r_{P},\quad r_P = 2\tanh^{-1}(r) = \ln\frac{1+r}{1-r} = \ln(1+\frac{2r}{1-r})

    \delta_{\text{rel}} = \frac{\ln(1+\sqrt{2})}{\ln(1+\frac{2r}{1-r})}
    $$
    """
    r = 1.0 - tol
    return np.log1p(np.sqrt(2)) / np.log1p(2 * r / (1 - r))
