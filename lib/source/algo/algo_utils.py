import math
import time

import numpy as np
from numba import cuda, jit, njit, prange
from line_profiler import profile

def calculate_frequencies(rows, cols, dict_freq):
    for i in range(rows.shape[0]):
      for j in range(rows.shape[1]):
        tup = tuple((rows[i][j], cols[i][j]))
        dict_freq[tup] += 1
    return dict_freq


@njit(parallel=True)
def prepare_batch_indices_flat(far_away_pairs, start_ind, end_ind, shape):
    batch_indices = np.empty(int(end_ind - start_ind) * 6, dtype=np.int32)


    for indx in prange(start_ind, end_ind):
        i, j = indx_to_2d(indx)

        pair_1 = far_away_pairs[i]
        pair_2 = far_away_pairs[j]

        batch_indices[6 * (indx-start_ind)] = pair_1[0]*shape[1] + pair_1[1]
        batch_indices[6 * (indx-start_ind) + 1] = pair_2[0]*shape[1] + pair_2[1]
        batch_indices[6 * (indx-start_ind) + 2] = pair_1[0]*shape[1] + pair_2[0]
        batch_indices[6 * (indx-start_ind) + 3] = pair_1[1]*shape[1] + pair_2[1]
        batch_indices[6 * (indx-start_ind) + 3] = pair_1[0]*shape[1] + pair_2[1]
        batch_indices[6 * (indx-start_ind) + 4] = pair_1[1]*shape[1] + pair_2[0]
        # batch_indices_row[indx - start_ind] = np.array(
        #     [pair_1[0], pair_2[0], pair_1[0], pair_1[1], pair_1[0], pair_1[1]],
        #     dtype=np.int32,
        # )

        # batch_indices_col[indx - start_ind] = np.array(
        #     [pair_1[1], pair_2[1], pair_2[0], pair_2[1], pair_2[1], pair_2[0]],
        #     dtype=np.int32,
        # )

    return batch_indices

def rows_cols_vals(shape, rows, cols, d_freq):
    true_rows = np.arange(shape[0])
    true_cols = np.arange(shape[1])
    values = np.zeros(len(true_rows))
    for i in range(true_rows):  
        if (true_rows[i], true_cols[i]) in d_freq.keys():
            values[i] = d_freq(true_rows[i], true_cols[i])
        else:
           values[i] = 0
    return true_rows, true_cols, values


def add_data(csv_path, dict_vals):
    """Adds data to csv file."""
    new_rows = []
    new_rows.append(dict_vals.values())
    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(new_rows)


def get_far_away_pairs(A, N):
    a = -A.ravel()
    a_indx = np.argpartition(a, N)
    indx_sorted = zip(
        *np.unravel_index(sorted(a_indx[: N + 1], key=lambda i: a[i]), A.shape)
    )
    return [(i, j) for (i, j) in indx_sorted if i < j]


def matrix_to_triangular(arr):
    triangular_part = []
    for i in range(arr.shape[0]):
        cur_list = []
        for j in range(i):
            cur_list.append(arr[i][j])
        triangular_part.append(np.asarray(cur_list))
    return np.asarray(triangular_part, dtype="object")


def cuda_prep_cartesian(dist_array, block_size):

    # threadsperblock = (block_size, block_size)
    threadsperblock = block_size
    
    # print(len(dist_array))
    

    blockspergrid = min(65535, int(dist_array.shape[0] / threadsperblock) + 1)
    # blockspergrid_y = min(
    #     65535, int(np.ceil(dist_array.shape[0] / threadsperblock[1])) + 1
    # )

    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    # blockspergrid = blockspergrid_x
    cartesian_dist_array = cuda.to_device(np.asarray(dist_array))
    delta_res = cuda.to_device(np.zeros(1))
    # deltas_res = cuda.device_array(cartesian_dist_array.shape[0])
    return cartesian_dist_array, delta_res, threadsperblock, blockspergrid


def cuda_prep_true_delta(dist_matrix):
    k_1 = int(math.log2(dist_matrix.shape[0])) + 1
    k_2 = 2 ** k_1 - 1
    k_3 = dist_matrix.shape[0]
    adj_m = cuda.to_device(dist_matrix.flatten()) 

    k = cuda.to_device([k_2, k_1, k_3])
    delta_res = cuda.to_device(np.zeros(1))

    print([k_2, k_1, k_3])


    block_size = (16, 16, 4)
    threadsperblock = block_size
    # np.int64(f"{dist_matrix.shape[0]}{dist_matrix.shape[0]}")
    blockspergrid_x = min(65535, int(np.ceil(dist_matrix.shape[0] ** 2 / threadsperblock[0])) + 1)
    blockspergrid_y = min(65535, int(np.ceil(dist_matrix.shape[0] / threadsperblock[1])) + 1)
    blockspergrid_z = min(65535, int(np.ceil(dist_matrix.shape[0] / threadsperblock[2])) + 1)

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    return adj_m, k, delta_res, threadsperblock, blockspergrid


# def cuda_prep_cartesian(dist_array, block_size):
#     threadsperblock = (block_size, block_size)
#     blockspergrid_x = min(
#         65535, int(np.ceil(dist_array.shape[0] / threadsperblock[0])) + 1
#     )
#     blockspergrid_y = min(
#         65535, int(np.ceil(dist_array.shape[0] / threadsperblock[1])) + 1
#     )
#     blockspergrid = (blockspergrid_x, blockspergrid_y)
#     cartesian_dist_array = cuda.to_device(np.asarray(dist_array))
#     deltas_res = cuda.device_array(cartesian_dist_array.shape[0])
#     return cartesian_dist_array, deltas_res, threadsperblock, blockspergrid


def cuda_prep(far_away_pairs, dist_matrix, block_size):
    x_coords, y_coords = (
        list(zip(*far_away_pairs))[0],
        list(zip(*far_away_pairs))[1],
    )
    x_coords = np.asarray(x_coords).astype(int)
    y_coords = np.asarray(y_coords).astype(int)

    x_coord_pairs = cuda.to_device(x_coords)
    y_coord_pairs = cuda.to_device(y_coords)
    # dist_matrix = matrix_to_triangular(dist_matrix)
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
        return max_workers_theory


def calc_max_lines(gpu_mem_bound, pairs_len):
    cartesian_size = int(pairs_len * (pairs_len - 1) / 2)
    parts = (cartesian_size * 6 * 8) / (gpu_mem_bound * math.pow(10, 9))
    print(cartesian_size)
    print(parts)
    max_lines = int(cartesian_size // parts)
    return max_lines


@njit()
def indx_to_2d(indx):
    n = round(math.sqrt(2 * indx))
    S_n = (1 + n) / 2 * n
    return n, int(n - (S_n - indx) - 1)


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


@njit(nopython=False)
def permutations(A, k):
    r = [[i for i in range(0)]]
    for i in range(k):
        r = [[a] + b for a in A for b in r if (a in b) == False]
    return r


@njit(nopython=False)
def combinations(A, k):
    return [item for item in permutations(A, k) if sorted(item) == item]


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
