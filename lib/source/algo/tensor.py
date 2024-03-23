from protes import protes


def delta_execution(dist_matrix):
    def one_try_delta(i, j, k):
        """

        Function for delta computation.

        # Parameters
        # ----------
        # i : int
        #   Index of first point to consider.

        # j : int
        #   Index of second point to consider.

        # k : int
        #   Index of third point to consider.

        Returns
        ----------
        delta : int
          A delta value for points of passed indices. Should be maximized.

        Notes
        ----------
        Function calculates semi-difference of the first and second maximums (m_1 and m_2) of the distances sums,
        which is declared to be an equivalent definition of Gromov`s delta according to the article https://inria.hal.science/hal-01199860/document.

        """
        sum_1 = dist_matrix[i][j] + dist_matrix[0][k]
        sum_2 = dist_matrix[i][k] + dist_matrix[0][j]
        sum_3 = dist_matrix[j][k] + dist_matrix[0][i]

        dist_array = [sum_1, sum_2, sum_3]
        m_1, m_2 = sorted(dist_array)[-2:]

        delta = (m_1 - m_2) / 2
        return delta

    return one_try_delta


def delta_protes(dist_matrix):
    delta_exe_func = delta_execution(dist_matrix)

    def call_delta(I):
        curr_values = []

        for mult_indx in I:
            curr_values.append(delta_exe_func(*mult_indx))
        return curr_values

    return call_delta


def tensor_approximation(d, b_s, func):
    """
    Function for comparing delta, using protes model of tensor approximation,
    visit https://github.com/AndreiChertkov/teneva for more details.

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
    f_batch = lambda I: func(I)
    _, y_opt = protes(
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
