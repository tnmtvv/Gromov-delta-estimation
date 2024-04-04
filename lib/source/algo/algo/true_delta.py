import numpy as np
from numba import cuda, jit, njit, prange

from line_profiler import profile
import math

@profile
def delta_hyp(dismat: np.ndarray) -> float:
    """
    Computes Gromov's delta-hyperbolicity value from distance matrix using the maxmin product.

    Parameters:
    -----------
    dismat : numpy.ndarray
        A square distance matrix of shape (n, n), where n is the number of nodes in a dataset.

    Returns:
    --------
    float
        The delta hyperbolicity value.

    Notes:
    ------
    This is a naive implementation that can be very inefficient on large datasets.
    Use other mode for efficient implementation.
    """
    deltas = np.zeros(dismat.shape[0])
    for p in range(dismat.shape[0]):
        # p = 0  # fixed point (hence the dataset should be shuffled for more reliable results)
        row = dismat[p, :][np.newaxis, :]
        col = dismat[:, p][:, np.newaxis]
        XY_p = 0.5 * (row + col - dismat)

        mins = np.minimum(XY_p[:, :, None], XY_p[None, :, :])
        maxmin = np.max(mins, axis=1)
        deltas[p] = np.max(maxmin - XY_p)
    return np.max(deltas)


@njit(parallel=True)
def true_delta_func(data):
    dim = np.shape(data)[0]
    delta = 0

    for p in prange(dim):
      for i in prange(dim):
        for j in prange(dim):
          g = 0.5 * (data[i][p] + data[p][j] - data[i][j])
          for k in prange(dim):
              b = 0.5  * (data[p][k] + min(data[i][p] - data[i][k], data[p][j] - data[k, j]))
              delta = max(delta, b - g)
    return delta


@cuda.jit
def true_delta_gpu(dismat: np.ndarray, delta_res: np.ndarray, num_arr: np.ndarray):
    # p, c, h = cuda.grid(3)
    num_1 = num_arr[0]
    num_2 = num_arr[1]
    d_m_len = num_arr[2]

    h = (cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x) & num_1
    k = (cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x) >> num_2
    # h = (cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x) // dismat.shape[0]
    # k = (cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x) % dismat.shape[0]
    c = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    p = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
    

    # h = h // dismat.shape[0]
    # k = h % dismat.shape[0]

    if p < d_m_len and c < d_m_len and h < d_m_len and k < d_m_len:
      cuda.atomic.max(
          delta_res, (0), 0.5 * (dismat[p * d_m_len +  h] + dismat[k * d_m_len + c] - max(dismat[p * d_m_len +  k] + dismat[h * d_m_len + c],  dismat[p * d_m_len + c] + dismat[k * d_m_len + h]))
      )
