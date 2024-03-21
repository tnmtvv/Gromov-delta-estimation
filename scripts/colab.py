from numba import njit, jit, prange, set_num_threads, typed, cuda
import numpy as np
import sys
import math
from timeit import default_timer as timer
import random

np.set_printoptions(threshold=sys.maxsize)


@njit()
def indx_to_2d(indx):
    n = round(math.sqrt(2 * indx))
    S_n = (1 + n) / 2 * n
    return n, int(n - (S_n - indx) - 1)


def build_dist_matrix(data):
    arr_all_dist = []
    for p in data:
        arr_dist = list(
            map(lambda x: 0 if (p == x).all() else np.linalg.norm(p - x), data)
        )
        arr_all_dist.append(arr_dist)
    arr_all_dist = np.asarray(arr_all_dist)
    return arr_all_dist


def get_far_away_pairs(A, N):
    a = -A.ravel()
    a_indx = np.argpartition(a, N)
    indx_sorted = zip(
        *np.unravel_index(sorted(a_indx[: N + 1], key=lambda i: a[i]), A.shape)
    )
    return [(i, j) for (i, j) in indx_sorted if i < j]


def generate_synthetic_points(dimensions, num_points):
    points = np.random.rand(num_points, dimensions)
    return points


def prepare_batch_indices(far_away_pairs, start_ind, end_ind):
    batch_indices_row = np.empty((int(end_ind - start_ind), 6), dtype=np.int32)
    batch_indices_col = np.empty((int(end_ind - start_ind), 6), dtype=np.int32)
    for indx in prange(start_ind, end_ind):
        i, j = indx_to_2d(indx)
        pair_1 = far_away_pairs[i]
        pair_2 = far_away_pairs[j]
        batch_indices_row[indx - start_ind] = np.array(
            [pair_1[0], pair_2[0], pair_1[0], pair_1[1], pair_1[0], pair_1[1]],
            dtype=np.int32,
        )
        batch_indices_col[indx - start_ind] = np.array(
            [pair_1[1], pair_2[1], pair_2[0], pair_2[1], pair_2[1], pair_2[0]],
            dtype=np.int32,
        )

    return batch_indices_row, batch_indices_col


def generate_indices(num, dim):
    return [
        (random.randint(0, dim - 1), random.randint(0, dim - 1)) for _ in range(num)
    ]


def experiments():
    dims = [100, 500, 1000, 5000]

    times = []
    for dim in dims:
        point_matr = generate_synthetic_points(3, dim)
        dist_matrix = build_dist_matrix(point_matr)
        del point_matr

        indices = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 20)
        print(len(indices))
        rows, cols = prepare_batch_indices(indices, 0, len(indices))
        time_start = timer()
        batch = dist_matrix[
            rows.reshape(-1),
            cols.reshape(-1),
        ].reshape(-1, 6)
        overall_time = timer() - time_start
        times.append(overall_time)

        del dist_matrix
        del indices
        print(f"done {dim}")

    print(times)


if __name__ == "__main__":
    experiments()
