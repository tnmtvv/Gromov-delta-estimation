import numpy as np
from numba import jit, cuda, njit, prange
import time
import math


def get_far_away_pairs(A, N):
    # a = zip(
    #     *np.unravel_index(np.argsort(-A.ravel())[:N], A.shape)
    # )  # переписать через партишн
    # return [(i, j) for (i, j) in a if i < j]
    a = -A.ravel()
    a_indx = np.argpartition(a, N)
    indx_sorted = zip(
        *np.unravel_index(sorted(a_indx[: N + 1], key=lambda i: a[i]), A.shape)
    )
    return [(i, j) for (i, j) in indx_sorted if i < j]


def cuda_prep(far_away_pairs, dist_matrix, block_size):
    x_coords, y_coords = (
        list(zip(*far_away_pairs))[0],
        list(zip(*far_away_pairs))[1],
    )
    x_coord_pairs = cuda.to_device(x_coords)
    y_coord_pairs = cuda.to_device(y_coords)
    adj_m = cuda.to_device(dist_matrix)
    # results = cuda.to_device(list(np.zeros(len(x_coord_pairs))))
    delta_res = cuda.to_device(list(np.zeros(1)))
    n = len(x_coord_pairs)

    threadsperblock = (block_size, block_size)
    blockspergrid_x = int(np.ceil(n / threadsperblock[0])) + 1
    blockspergrid_y = int(np.ceil(n / threadsperblock[1])) + 1
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return (
        n,
        x_coord_pairs,
        y_coord_pairs,
        adj_m,
        # results,
        blockspergrid,
        threadsperblock,
        delta_res,
    )


def calc_max_workers(batch_size, mem_bound, n_tries):
    matrix_size_gb = (batch_size * batch_size * 8) / math.pow(10, 9)
    max_workers_theory = int(mem_bound // (matrix_size_gb * 2))
    if max_workers_theory > n_tries:
        return n_tries
    else:  # think about the situation, where the matrix size itself is bigger than memory bound (max_workers_thery < 1)
        return max(1, max_workers_theory)


@njit(parallel=True)
def s_delta_parallel(far_away_pairs, A, pid, x, y, h_lb):
    delta_hyp = np.zeros(pid, dtype=A.dtype)
    for inx in prange(pid):
        v = far_away_pairs[inx][0]
        w = far_away_pairs[inx][1]

        S1 = A[x, y] + A[v, w]
        S2 = A[x, v] + A[y, w]
        S3 = A[x, w] + A[y, v]
        delta_hyp[inx] = S1 - max(S2, S3)
    return np.max(delta_hyp)


@njit
def s_delta(dist, ind_i, ind_j, k, delta_hyp_k):
    dist_0k = dist[0][k - 1]
    dist_0i = dist[0][ind_i]
    dist_ik = dist[ind_i][k - 1]

    dist_0j = dist[0][ind_j]
    dist_jk = dist[ind_j][k - 1]
    dist_ij = dist[ind_i][ind_j]

    # algo with S
    dist_array = [dist_0j + dist_ik, dist_0i + dist_jk, dist_0k + dist_ij]
    s2, s1 = sorted(dist_array)[-2:]
    delta_hyp_k = max(delta_hyp_k, s1 - s2)
    return max(delta_hyp_k, s1 - s2)


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
