import numpy as np
from numba import njit, prange


@njit(parallel=True)
def delta_hyp_condensed_rand_top(
    dist: np.ndarray, n_samples, const, mode="top_k"
) -> float:
    """
    Compute the delta hyperbolicity value from the condensed distance matrix representation.
    This is a modified version of the `delta_hyp_condenced` function.

    Parameters
    ----------
    dist_condensed : numpy.ndarray
        A 1D array representing the condensed distance matrix of the dataset.
    n_samples : int
        The number of nodes in the dataset.
    const : int
        Number of most distant points that are conciedered by the algo.
    const : str
        Mode offunction execution.

    Returns
    -------
    float
        The delta hyperbolicity of the dataset.

    Notes
    -----
    The idea is that we can select points partly randomly to achieve a better covering of an item space.
    """
    delta_hyp = np.zeros(n_samples, dtype=dist.dtype)

    for k in prange(n_samples):
        # as in `delta_hyp`, fixed point is selected at 0
        delta_hyp_k = 0.0
        dist_0k = dist[0][k - 1]

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

        for ind_i in considered_i:
            dist_0i = dist[0][ind_i]
            dist_ik = dist[ind_i][k - 1]

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
                dist_0j = dist[0][ind_j]
                dist_jk = dist[ind_j][k - 1]
                dist_ij = dist[ind_i][ind_j]

                dist_array = [dist_0j + dist_ik, dist_0i + dist_jk, dist_0k + dist_ij]
                s1 = max(dist_array)
                dist_array.remove(s1)
                s2 = max(dist_array)
                delta_hyp_k = max(delta_hyp_k, s1 - s2)
        delta_hyp[k] = delta_hyp_k
    return 0.5 * np.max(delta_hyp)


@njit(parallel=True)
def delta_hyp_condensed(dist_condensed: np.ndarray, n_samples: int) -> float:
    """
    Compute the delta hyperbolicity value from the condensed distance matrix representation.
    This is a more efficient analog of the `delta_hyp` function.

    Parameters
    ----------
    dist_condensed : numpy.ndarray
        A 1D array representing the condensed distance matrix of the dataset.
    n_samples : int
        The number of nodes in the dataset.

    Returns
    -------
    float
        The delta hyperbolicity of the dataset.

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
def delta_hyp_condensed_new(
    dist_condensed: np.ndarray, n_samples: int, const: int
) -> float:
    """
    Compute the delta hyperbolicity value from the condensed distance matrix representation.
    This is a more efficient analog of the `delta_hyp_condensed` function.

    Parameters
    ----------
    dist_condensed : numpy.ndarray
        A 1D array representing the condensed distance matrix of the dataset.
    n_samples : int
        The number of nodes in the dataset.
    const : int
        Number of most distant points that are conciedered by the algo.

    Returns
    -------
    float
        The delta hyperbolicity of the dataset.

    Notes
    -----
    Heuristic version of delta_hyp_condenced function. Attemp to apply CCL main idea to the condenced implementation.
    """
    delta_hyp = np.zeros(n_samples, dtype=dist_condensed.dtype)
    seen = np.array([0, 0, 0])

    for k in prange(n_samples):
        # as in `delta_hyp`, fixed point is selected at 0
        delta_hyp_k = 0.0
        dist_0k = dist_condensed[0][k - 1]

        # sorting distances from k to all other points, index i will be chosen from the most distant ones
        inds_i = np.argpartition(dist_condensed[k - 1], n_samples - const)
        considered_i = inds_i[-const:]
        for ind_i in considered_i:
            dist_0i = dist_condensed[0][ind_i]
            dist_ik = dist_condensed[ind_i][k - 1]

            # sorting distances from i to all other points, index j will be chosen from the most distant ones
            inds_j = np.argpartition(dist_condensed[:, ind_i], n_samples - (const + 1))
            considered_j = inds_j[-const:]

            for ind_j in considered_j:
                cur_indxs = np.asarray([k, ind_i, ind_j])
                np.append(seen, cur_indxs)
                dist_0j = dist_condensed[0][ind_j]
                dist_jk = dist_condensed[ind_j][k - 1]
                dist_ij = dist_condensed[ind_i][ind_j]

                # algo with S
                dist_array = [dist_0j + dist_ik, dist_0i + dist_jk, dist_0k + dist_ij]
                s1 = max(dist_array)
                dist_array.remove(s1)
                s2 = max(dist_array)
                delta_hyp_k = max(delta_hyp_k, s1 - s2)
        delta_hyp[k] = delta_hyp_k
    return 0.5 * np.max(delta_hyp)
