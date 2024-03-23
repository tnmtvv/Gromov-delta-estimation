import gc
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from timeit import default_timer as timer

import numpy as np
from lib.source.algo.algo_utils import (
    calc_max_workers,
)

from numba import cuda
from lib.source.algo.pipline_strategies import *

      
def batched_delta_hyp(
    X,
    strategy: PipelineStrategy,
    n_tries=10,
    batch_size=400,
    seed=42,
    max_workers=25,
    mem_cpu_bound=16,
    mem_gpu_bound=16,
    
):
    """
    Estimate the Gromov's delta hyperbolicity of a dataset using batch processing.

    Parameters
    ----------
    X : numpy.ndarray
        A 2D array of shape (n, m), where n is the number of nodes in a dataset and m is the dimensionality of the space
        that the nodes are embedded in.
    n_tries : int, optional
        The number of times to compute the delta hyperbolicity using different subsets of nodes.
    batch_size : int, optional
        The number of nodes to process in each batch.
    seed : int or None, optional
        Seed used for the random generator in batch sampling. Default is 42.
    economic : bool, optional
        If True, the function will use more memory-efficient methods. Default is True.
    max_workers : int or None, optional
        The maximum number of workers to use. If None, the number will be set to the number of available CPUs. Default is None.
    way : string
        Mode for calculations.

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples containing the delta hyperbolicity and diameter values of the dataset for different batches.

    Notes
    -----
    The function computes the delta hyperbolicity value of a dataset using batch processing.
    For each batch of nodes, the function computes the pairwise distances, computes the delta hyperbolicity value,
    and then aggregates the results across all batches to obtain the final delta hyperbolicity value.
    If economic=True, the function will use more efficient version of delta_hyp to combat better complexity.
    Pass way parameter to choose the mode.
    """
    print("true X shape" + str(X.shape))
    results = []
    
    max_workers_cpu = min(
        max_workers, calc_max_workers(batch_size, mem_cpu_bound, n_tries)
    )
    if max_workers_cpu < 1:
        max_workers_cpu = 1
    max_workers_gpu = min(
        max_workers, calc_max_workers(batch_size, mem_gpu_bound, n_tries)
    )
    if isinstance(strategy, SeparateCartesianStrategy) and max_workers_gpu >= 1:
        strategy = SeparateStrategy(l_multiple=strategy.l, max_workers_gpu=max_workers_gpu)
        x = cuda.device_array(1)  # resolving strange numba error
    
    results = strategy.pipeline(X=X, max_workers_cpu=max_workers_cpu, n_tries=n_tries, batch_size=batch_size, seed=seed)
    return results
