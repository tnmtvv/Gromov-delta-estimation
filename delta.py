import numpy as np
import time
import typing

from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix

from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances
from hyplearn import hyp_learn_euclidean_distances

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from timeit import default_timer as timer

# import cython
# import sage.all
# from sage.graphs.hyperbolicity import hyperbolicity_BCCM

from scipy.spatial.distance import pdist, cdist, squareform
from fastdist import fastdist
from numba import njit, prange, set_num_threads, typed

# from Opti import Target
# from teneva_opti import *

from protes import protes

# try:
#     import networkx as nx
# except ImportError:
#     nx = None


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


@njit(parallel=True)
def delta_hyp_condensed_CCL(far_apart_pairs: np.ndarray, adj_m: np.ndarray):
    n_samples = adj_m.shape[0]
    # delta_hyp = np.zeros(n_samples, dtype=adj_m.dtype)
    delta_hyp = 0.0
    for i in prange(1, min(300000, len(far_apart_pairs))):
        pair_1 = far_apart_pairs[i]
        # if adj_m[pair_1[0]][pair_1[1]] < 2 * delta_hyp:
        #     return delta_hyp
        for j in prange(i):
            pair_2 = far_apart_pairs[j]
            if pair_2[0] not in pair_1 and pair_2[1] not in pair_1:
                i = pair_1[0]
                j = pair_1[1]
                v = pair_2[0]
                w = pair_2[1]

                d_ij = adj_m[i][j]
                d_iw = adj_m[i][w]
                d_iv = adj_m[i][v]

                d_jw = adj_m[j][w]
                d_jv = adj_m[j][v]

                d_vw = adj_m[v][w]

                cur_del = (d_ij + d_vw - max(d_iv + d_jw, d_iw + d_jv)) / 2
                delta_hyp = max(delta_hyp, cur_del)

    return delta_hyp


# @profile
def batched_delta_hyp(
    X,
    n_tries=10,
    batch_size=400,
    seed=42,
    economic=True,
    max_workers=25,
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
    way: string
        Mode for calculations.

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
    print("true X shape" + str(X.shape))
    n_objects, _ = X.shape
    # _, n_objects = X.shape

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
            # logger.info(f'Trial {i + 1}/{n_tries} relative delta: {delta_rel} for estimated diameter: {diam}')
            results.append(res)
    return results


# @profile
def delta_hyp_rel(X: np.ndarray, economic: bool = True, way="new"):
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
    # dist_matrix = hyp_learn_euclidean_distances(X, triangular=True)
    dist_matrix = pairwise_distances(X, metric="euclidean")
    # print(X.shape)
    # dist_matrix = squareform(dist_condensed)
    diam = np.max(dist_matrix)

    if economic:
        if way == "heuristic":
            far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 20)
            print("true: " + str(len(far_away_pairs)))
            delta = delta_Nastyas(dist_matrix, far_away_pairs, 100000)
        elif way == "CCL":
            print("ccl")
            far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 20)
            delta = delta_hyp_condensed_CCL(far_away_pairs, dist_matrix)
        elif way == "article":
            delta = delta_hyp_condensed_article(
                dist_matrix, k=(dist_matrix.shape[0] * 30) // 100
            )
        elif way == "rand_top":
            const = min(50, X.shape[0] - 1)
            delta = delta_hyp_condensed_rand_top(
                dist_matrix, X.shape[0], const, mode="top_rand"
            )
        elif way == "tensor":
            # used_indices = []
            objective_func = delta_protes(dist_matrix)
            delta = tensor_approximation(
                d=3, b_s=dist_matrix.shape[0], func=objective_func
            )
    delta_rel = 2 * delta / diam
    return delta_rel, diam


def delta_protes(dist_matrix):
    delta_exe_func = delta_execution(dist_matrix)

    def call_delta(I):
        curr_values = []

        for mult_indx in I:
            curr_values.append(delta_exe_func(*mult_indx))
        return curr_values

    return call_delta


def delta_execution(dist_matrix):
    def one_try_delta(i, j, k):
        """

        Function for delta computation.

        # Parameters
        # ----------
        # i : int
        #   Index of first point to concieder.

        # j : int
        #   Index of second point to concieder.

        # k : int
        #   Index of third point to concieder.

        Returns
        ----------
        delta : int
          A delta value for points of passed indices. Should be maximaized.

        Notes
        ----------
        Function calculates semi-difference of the first and second maximums (m_1 and m_2) of the distances sums,
        whith declared to be an eqvivalent deffinition of Gromov`s delta according to the article https://inria.hal.science/hal-01199860/document.

        """
        sum_1 = dist_matrix[i][j] + dist_matrix[0][k]
        sum_2 = dist_matrix[i][k] + dist_matrix[0][j]
        sum_3 = dist_matrix[j][k] + dist_matrix[0][i]

        dist_array = [sum_1, sum_2, sum_3]
        m_1 = max(dist_array)
        dist_array.remove(m_1)
        m_2 = max(dist_array)

        delta = (m_1 - m_2) / 2
        return delta

    return one_try_delta


def time_func(func):
    def res_func(*args, **kwargs):
        time_start = timer()
        res = func(*args, **kwargs)
        time_finish = timer() - time_start
        print(time_finish)
        return res

    return res_func


def compare(
    delta_func_1, delta_func_2, precision: bool, way, economic, max_workers, **kwargs
):
    delta_1_start = timer()
    delta_1 = delta_func_1(
        max_workers=max_workers, economic=economic, way=way, **kwargs
    )
    delta_1_time = timer() - delta_1_start

    delta_2_start = timer()
    delta_2 = delta_func_2(**kwargs)
    delta_2_time = timer() - delta_2_start

    if precision:
        try:
            np.testing.assert_allclose(delta_1, delta_2, rtol=1e-2, atol=0)
        except AssertionError:
            print("Deltas do not match")

    print("time :")
    print("delta_1 time: " + str(delta_1_time))
    print("delta_2 time: " + str(delta_2_time))


def deltas_comparison(
    X,
    n_tries=10,
    batch_size=400,
    seed=42,
    max_workers=25,
    rank=10,
):
    rel_delta_start = timer()
    deltas_diams = batched_delta_hyp(
        X,
        n_tries=n_tries,
        batch_size=batch_size,
        seed=seed,
        economic=True,
        max_workers=max_workers,
        rank=rank,
        way="old",
        approach="old",
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
        rank=rank,
        way="old",
        approach="old",
    )
    true_deltas = list(map(lambda x: x[0], true_delta))
    true_delta = np.mean(true_deltas)

    true_delta_time = timer() - true_delta_start

    # tensor_delta_start = timer()
    # tensor_delta = tensor_approximation()
    # tensor_delta_time = timer() - tensor_delta_start

    print("---------------------------")
    print("true_delta " + str(true_delta))
    print("rel_delta " + str(rel_delta))
    # print('tensor_delta ' + str(tensor_delta))
    print()
    print("true_delta time " + str(true_delta_time))
    print("rel_delta time " + str(rel_delta_time))
    # print('tensor_delta time ' + str(tensor_delta_time))
    print("---------------------------")


def tensor_approximation(d, b_s, func):
    f_batch = lambda I: func(I)
    i_opt, y_opt = protes(
        f=f_batch,
        d=d,
        k=1000,
        n=b_s,
        m=1.0e7,
        k_top=50,
        log=False,
        is_max=True,
        r=7,
        lr=5.0e-1,
        k_gd=2,
    )
    return y_opt


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
def delta_hyp_condensed_rand_top(
    dist: np.ndarray, n_samples, const, mode="rand"
) -> float:
    delta_hyp = np.zeros(n_samples, dtype=dist.dtype)

    for k in prange(n_samples):
        # as in `delta_hyp`, fixed point is selected at 0
        delta_hyp_k = 0.0
        dist_0k = dist[0][k - 1]

        # сортим расстояния от k до остальных точек, индекс i будем выбирать из самых дальних

        if mode == "top_k":
            inds_i = np.argpartition(dist[k - 1], -const)
            considered_i = inds_i[-const:]
        elif mode == "top_rand":
            inds = np.argpartition(dist[k - 1], -const // 2)
            considered_i_top_part = inds[-const // 2 :]

            considered_i_rand_part = np.random.choice(inds[: -const // 2], const // 2)
            considered_i = np.concatenate(
                (considered_i_top_part, considered_i_rand_part)
            )
        else:
            considered_i = np.random.choice(n_samples, const)
        # брать 10 процентов от размера выборки ?

        for ind_i in considered_i:
            dist_0i = dist[0][ind_i]
            dist_ik = dist[ind_i][k - 1]
            # dist_condensed[ind_i][k - 1] = 0.0

            # сортим расстояния от i до остальных точек, индекс j будем выбирать из самых дальних

            if mode == "top_k":
                inds_j = np.argpartition(dist[ind_i - 1], -const)
                considered_j = inds_j[-const:]
            elif mode == "top_rand":
                inds = np.argpartition(dist[ind_i - 1], -const // 2)
                considered_j_top_part = inds[-const // 2 :]

                considered_j_rand_part = np.random.choice(
                    inds[: -const // 2], const // 2
                )
                considered_j = np.concatenate(
                    (considered_j_top_part, considered_j_rand_part)
                )
            else:
                considered_j = np.random.choice(n_samples, const)

            for ind_j in considered_j:
                cur_indxs = np.asarray([k, ind_i, ind_j])
                # np.append(seen, cur_indxs)
                dist_0j = dist[0][ind_j]
                dist_jk = dist[ind_j][k - 1]
                dist_ij = dist[ind_i][ind_j]

                dist_array = [dist_0j + dist_ik, dist_0i + dist_jk, dist_0k + dist_ij]
                s1 = max(dist_array)
                dist_array.remove(s1)
                s2 = max(dist_array)
                delta_hyp_k = max(delta_hyp_k, s1 - s2)
                # else:
                #     continue
        delta_hyp[k] = delta_hyp_k
    return 0.5 * np.max(delta_hyp)


@njit(parallel=True)
def delta_hyp_condensed_new(dist_condensed: np.ndarray, n_samples: int, const) -> float:
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
    seen = np.array([0, 0, 0])

    for k in prange(n_samples):
        # as in `delta_hyp`, fixed point is selected at 0
        delta_hyp_k = 0.0
        dist_0k = dist_condensed[0][k - 1]

        # сортим расстояния от k до остальных точек, индекс i будем выбирать из самых дальних
        inds_i = np.argpartition(dist_condensed[k - 1], n_samples - const)
        considered_i = inds_i[-const:]
        # considered_i = considered_i[considered_i != k]

        # брать 10 процентов от размера выборки ?
        for ind_i in considered_i:
            dist_0i = dist_condensed[0][ind_i]
            dist_ik = dist_condensed[ind_i][k - 1]

            # сортим расстояния от i до остальных точек, индекс j будем выбирать из самых дальних
            inds_j = np.argpartition(dist_condensed[:, ind_i], n_samples - (const + 1))
            considered_j = inds_j[-const:]

            for ind_j in considered_j:
                cur_indxs = np.asarray([k, ind_i, ind_j])
                # if cur_indxs.all() not in seen:
                np.append(seen, cur_indxs)
                dist_0j = dist_condensed[0][ind_j]
                dist_jk = dist_condensed[ind_j][k - 1]
                dist_ij = dist_condensed[ind_i][ind_j]

                # алгоритм с S
                dist_array = [dist_0j + dist_ik, dist_0i + dist_jk, dist_0k + dist_ij]
                s1 = max(dist_array)
                dist_array.remove(s1)
                s2 = max(dist_array)
                delta_hyp_k = max(delta_hyp_k, s1 - s2)
        delta_hyp[k] = delta_hyp_k
    return 0.5 * np.max(delta_hyp)


def get_far_away_pairs(A, N):
    a = zip(*np.unravel_index(np.argsort(-A.ravel())[:N], A.shape))
    return [(i, j) for (i, j) in a if i < j]


@njit(paralell=True)
def delta_Nastyas(A, far_away_pairs, i_break=50000):
    h_lb = 0
    h_ub = np.inf

    for pid in prange(min(i_break, len(far_away_pairs))):
        if pid >= i_break:
            break

        p = far_away_pairs[pid]
        x, y = p

        dist = A[p]

        if dist < h_ub:
            if h_ub <= h_lb:
                break

            h_ub = dist

        if dist <= h_lb:
            # delta всегда меньше чем минимальное расстояние в четверке точек, если мы будем рассматривать четверки с текущим dist, то знаем, что дельту больше dist мы уже не получим,
            # значит h_lb не обновится, так что можно остановиться
            break

        for inx in prange(pid):
            v = far_away_pairs[inx][0]
            w = far_away_pairs[inx][1]

            S1 = A[x, y] + A[v, w]
            S2 = A[x, v] + A[y, w]
            S3 = A[x, w] + A[y, v]
            h_lb = max(h_lb, S1 - max(S2, S3))

            if h_ub == h_lb:
                break

    return h_lb / 2


@njit
def set_intersect(lst1, lst2):
    return set(lst1).intersection(set(lst2))


@njit
def loop_intersection(lst1, lst2):
    result = []
    for element1 in lst1:
        for element2 in lst2:
            if element1 == element2:
                result.append(element1)
    return result


# @profile
@njit(parallel=True)
def delta_hyp_condensed_article(dist_condensed: np.ndarray, k: int) -> float:
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
    delta_hyp = 0

    n_samples = dist_condensed.shape[0]
    x = np.random.randint(1, n_samples)

    indx_a = np.argmax(dist_condensed[x - 1])

    indx_b = np.argmax(dist_condensed[indx_a])
    dist_a_b = dist_condensed[indx_a][indx_b]

    S_a = np.where(dist_condensed[indx_a] >= dist_a_b / 2)[0]
    S_b = np.where(dist_condensed[indx_b] >= dist_a_b / 2)[0]
    # print("S_a")
    # print(S_a)

    c_indxs = loop_intersection(typed.List(S_a), typed.List(S_b))[:k]
    for i in prange(len(c_indxs)):
        indx_c = c_indxs[i]
        dist_a_c = dist_condensed[indx_a][indx_c]
        dist_b_c = dist_condensed[indx_b][indx_c]
        for indx_d in prange(n_samples):
            # if indx_d != indx_a and indx_d != indx_b:
            dist_a_d = dist_condensed[indx_a][indx_d]
            dist_b_d = dist_condensed[indx_b][indx_d]
            dist_c_d = dist_condensed[indx_c][indx_d]
            dist_array = [
                dist_a_b + dist_c_d,
                dist_a_c + dist_b_d,
                dist_a_d + dist_b_c,
            ]
            s1 = max(dist_array)
            dist_array.remove(s1)
            s2 = max(dist_array)
            delta_hyp = max(delta_hyp, s1 - s2)
    return 0.5 * delta_hyp


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
