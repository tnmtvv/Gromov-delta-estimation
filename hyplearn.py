import numpy as np
from numba import njit, prange, set_num_threads, typed
from scipy.linalg.blas import dsyrk, ssyrk


@njit(fastmath=True, nogil=True, cache=True, parallel=True)
def _maximum0(XXT, squared=True):
    """
    [Added 15/10/2018] [Edited 21/10/2018]
    Computes maxmimum(XXT, 0) faster. Much faster than Sklearn since uses
    the notion that	distance(X, X) is symmetric.

    Steps:
            maximum(XXT, 0)
                    Optimised. Instead of n^2 operations, does n(n-1)/2 operations.
    """
    n = len(XXT)

    for i in prange(n):
        XXT[i, i] = 0
        for j in prange(i):
            if XXT[i, j] < 0:
                XXT[i, j] = 0
            if not squared:
                XXT[i, j] **= 0.5

    return XXT


# maximum0_parallel = njit(_maximum0, fastmath=True, nogil=True, parallel=True)


@njit(fastmath=True, nogil=True, cache=True, parallel=True)
def _reflect(X):
    """
    See reflect(X, n_jobs = N) documentation.
    """
    n = len(X)
    for i in prange(1, n):
        Xi = X[i]
        for j in range(i):
            X[j, i] = Xi[j]
    return X


# reflect_single = njit(_reflect, fastmath=True, nogil=True, cache=True)
# reflect_parallel = njit(_reflect, fastmath=True, nogil=True, parallel=True)


@njit()
def reflect(X, n_jobs=1):
    """
    [Added 15/10/2018] [Edited 18/10/2018]
    Reflects lower triangular of matrix efficiently to upper.
    Notice much faster than say X += X.T or naive:
            for i in range(n):
                    for j in range(i, n):
                            X[i,j] = X[j,i]
    In fact, it is much faster to perform vertically:
            for i in range(1, n):
                    Xi = X[i]
                    for j in range(i):
                            X[j,i] = Xi[j]
    The trick is to notice X[i], which reduces array access.
    """
    X = _reflect(X)
    return X


@njit(fastmath=True, nogil=True, cache=True, parallel=True)
def mult_minus2(XXT):
    """
    [Added 17/10/2018]
    Quickly multiplies XXT by -2. Uses notion that XXT is symmetric,
    hence only lower triangular is multiplied.
    """
    n = len(XXT)
    for i in prange(n):
        for j in range(i):
            XXT[i, j] *= -2
    return XXT


@njit(fastmath=True, nogil=True, cache=True, parallel=True)
def rowSum_A(X, norm=False):
    """
    [Added 22/10/2018]
    Computes rowSum**2 for dense array efficiently, instead of using einsum
    """
    n = len(X)
    s = 0
    for i in prange(n):
        s += X[i] ** 2
    if norm:
        s **= 0.5
    return s


@njit(fastmath=True, nogil=True, cache=True, parallel=True)
def rowSum_0(X, norm=False):
    """
    [Added 17/10/2018]
    Computes rowSum**2 for dense matrix efficiently, instead of using einsum
    """
    n, p = X.shape
    S = np.zeros(n, dtype=X.dtype)

    for i in prange(n):
        s = 0
        Xi = X[i]
        for j in prange(p):
            Xij = Xi[j]
            s += Xij * Xij
        S[i] = s
    if norm:
        S **= 0.5
    return S


@njit()
def rowSum(X, norm=False):
    """
    [Added 22/10/2018]
    Combines rowSum for matrices and arrays.
    """

    return rowSum_0(X, norm)
    # return rowSum_A(X, norm)


def _XXT(XT):
    """
    [Added 30/9/2018]
    Computes X @ XT much faster than naive X @ XT.
    Notice X @ XT is symmetric, hence instead of doing the
    full matrix multiplication X @ XT which takes O(pn^2) time,
    compute only the upper triangular which takes slightly
    less time and memory.
    """
    if XT.dtype == np.float64:
        return dsyrk(1, XT, trans=1).T
    return ssyrk(1, XT, trans=1).T


def hyp_learn_euclidean_distances(
    X, Y=None, triangular=False, squared=False, n_jobs=-1
):
    """
    [Added 15/10/2018] [Edited 16/10/2018]
    [Edited 22/10/2018 Added Y option]
    Notice: parsing in Y will result in only 10% - 15% speed improvement, not 30%.

    Much much faster than Sklearn's implementation. Approx not 30% faster. Probably
    even faster if using n_jobs = -1. Uses the idea that distance(X, X) is symmetric,
    and thus algorithm runs only on 1/2 triangular part.

    Old complexity:
            X @ XT 			n^2p
            rowSum(X^2)		np
            XXT*-2			n^2
            XXT+X^2			2n^2
            maximum(XXT,0)	n^2
                                            n^2p + 4n^2 + np
    New complexity:
            sym X @ XT 		n^2p/2
            rowSum(X^2)		np
            sym XXT*-2		n^2/2
            sym XXT+X^2		n^2
            maximum(XXT,0)	n^2/2
                                            n^2p/2 + 2n^2 + np

    So New complexity approx= 1/2(Old complexity)
    """
    S = rowSum(X)
    if Y is X:
        # if X == Y, then defaults to fast triangular L2 distance algo
        Y = None

    if Y is None:
        XXT = _XXT(X.T)
        XXT = mult_minus2(XXT)

        XXT += S[:, np.newaxis]
        XXT += S  # [newaxis,:]

        D = _maximum0(XXT)
        if not triangular:
            D = reflect(XXT, n_jobs)
    else:
        D = X @ Y.T
        D *= -2
        D += S[:, np.newaxis]
        D += rowSum(Y)
        D = np.maximum(D, 0)
        if not squared:
            D **= 0.5
    return D
