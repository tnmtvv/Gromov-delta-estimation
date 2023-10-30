from numba import njit, prange, cuda
import numpy as np
from algo_utils import s_delta_parallel


@njit(parallel=True, fastmath=True)
def delta_hyp_condensed_CCL(far_apart_pairs: np.ndarray, adj_m: np.ndarray):
    """
    Computes Gromov's delta-hyperbolicity value with the basic approach, proposed in the article
    "On computing the Gromov hyperbolicity", 2015, by Nathann Cohen, David Coudert, Aurélien Lancin.

    Parameters:
    -----------
    far_apart_pairs : numpy.ndarray
        List of pairs of points, sorted by decrease of distance i.e. the most distant pair must be the first one.

    adj_m: numpy.ndarry
        Distance matrix.

    Returns:
    --------
    float
        The delta hyperbolicity value.
    """
    n_samples = adj_m.shape[0]
    delta_hyp = 0.0
    for i in prange(1, min(300000, len(far_apart_pairs))):
        pair_1 = far_apart_pairs[i]
        for j in prange(i):
            pair_2 = far_apart_pairs[j]
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


@cuda.jit
def delta_hyp_CCL_GPU(n, fisrt_points, second_points, adj_m, results):
    """
    Computes Gromov's delta-hyperbolicity value with the basic approach, proposed in the article
    "On computing the Gromov hyperbolicity", 2015, by Nathann Cohen, David Coudert, Aurélien Lancin.
    Algorithm was rewritten for execution on GPU.

    Parameters:
    -----------
    n: int
        The number of pairs.

    pairs_x_coord:
        List of the fisrt points of the far away pairs pairs.


    pairs_y_coord:
        List of the second points of the far away pairs pairs.


    adj_m: numpy.ndarry
        Distance matrix.

    x_coords_pairs
    far_apart_pairs: numpy.ndarray
        List of pairs of points, sorted by decrease of distance i.e. the most distant pair must be the first one.

    adj_m: numpy.ndarry
        Distance matrix.

    results:
        Array, where deltas for each pair will be stored.

    """
    n_samples = n
    idx = cuda.grid(1)
    if idx < n_samples:
        i = fisrt_points[idx]
        j = second_points[idx]

        results[idx] = 0
        delta_hyp = 0
        for k in range(idx):
            v = fisrt_points[k]
            w = second_points[k]

            d_ij = adj_m[i][j]
            d_iw = adj_m[i][w]
            d_iv = adj_m[i][v]

            d_jw = adj_m[j][w]
            d_jv = adj_m[j][v]

            d_vw = adj_m[v][w]

            cur_del = (d_ij + d_vw - max(d_iv + d_jw, d_iw + d_jv)) / 2
            delta_hyp = max(delta_hyp, cur_del)
        results[idx] = delta_hyp


def delta_CCL_heuristic(A, far_away_pairs, i_break=50000):
    """
    Version of CCL algo with iterations budjet.

    Parameters
    ----------
    dist_condensed : numpy.ndarray
        A 1D array representing the condensed distance matrix of the dataset.
    n_samples : int
        The number of nodes in the dataset.
    const : int
        Number of most distant points that are conciedered by the algo.
    i_break : int
        Max allowed iterations.

    Returns
    -------
    float
        The delta hyperbolicity of the dataset.

    Notes
    -----
    Heuristic version of delta_hyp_condenced function. Attemp to apply CCL main idea to the condenced implementation.
    """
    h_lb = 0
    h_ub = np.inf

    print(len(far_away_pairs))
    for pid in range(1, min(i_break, len(far_away_pairs))):
        p = far_away_pairs[pid]
        x, y = p
        h_lb = max(h_lb, s_delta_parallel(far_away_pairs, A, pid, x, y, h_lb))
    return h_lb / 2
