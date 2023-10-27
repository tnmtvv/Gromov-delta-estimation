import numpy as np


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

    p = 0  # fixed point (hence the dataset should be shuffled for more reliable results)
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)

    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)
