import numpy as np
from timeit import default_timer as timer
from delta import batched_delta_hyp
from numba import jit, cuda


@jit(fastmath=True)
def get_far_away_pairs(A, N):
    a = zip(*np.unravel_index(np.argsort(-A.ravel())[:N], A.shape))
    return [(i, j) for (i, j) in a if i < j]


def cuda_prep(far_away_pairs, dist_matrix, block_size):
    x_coords, y_coords = (
        list(zip(*far_away_pairs))[0],
        list(zip(*far_away_pairs))[1],
    )
    x_coord_pairs = cuda.to_device(x_coords)
    y_coord_pairs = cuda.to_device(y_coords)
    adj_m = cuda.to_device(dist_matrix)
    results = cuda.to_device(list(np.zeros(len(x_coord_pairs))))
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
        results,
        blockspergrid,
        threadsperblock,
    )


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
    way="heuristic",
):
    """
    Function for comparing delta, clculated with some heuristic method and ground truth delta (calculated with basic approach).

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
