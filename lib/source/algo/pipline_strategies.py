from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List
from lib.source.algo.delta_strategies import *

from lib.source.algo.algo_utils import calculate_matrices_pairs, calc_max_lines, prepare_batch_indices_flat, cuda_prep_cartesian
import cython_batching

class PipelineStrategy(ABC):
    @abstractmethod
    def pipeline(self, **kwargs) -> List[float]:
        pass

class UniteStrategy(PipelineStrategy):
    def __init__(self, strategy: DeltaStrategy):
        self.strategy = strategy
    def pipeline(self, X, max_workers_cpu, n_tries=10, batch_size=400, seed=42) -> List[float]:
        n_objects, _ = X.shape
        rng = np.random.default_rng(seed)
        results = []

        for part in range(int(n_tries // max_workers_cpu) + 1):
            if max_workers_cpu * part >= n_tries:
                break
            else:
                with ThreadPoolExecutor(
                    max_workers=min(n_tries - max_workers_cpu * part, max_workers_cpu)
                ) as executor:
                    futures = []
                    for _ in range(executor._max_workers):
                        if batch_size >= n_objects:
                            print("batch_size >= n_objects")
                            print(batch_size)
                            print(n_objects)
                            # `delta_hyp` selects a fixed point w.r.t which delta is computed
                            # the fixed point always corresponds to the first object, so shuffling allows
                            # exploring different initialization of a fixed point
                            batch_idx = rng.permutation(n_objects)
                        else:
                            batch_idx = rng.choice(
                                n_objects, batch_size, replace=False, shuffle=True
                            )
                        item_space = X[batch_idx]
                        print("batch done")
                        future = executor.submit(
                            self.strategy.calculate_delta, item_space
                        )
                        futures.append(future)
                    for i, future in enumerate(as_completed(futures)):
                        delta_rel, diam = res = future.result()
                        print("res: " + str(res))
                        results.append(res)
        return results

class SeparateStrategy():
    def __init__(self, l_multiple, max_workers_gpu):
        self.l = l_multiple
        self.max_workers_gpu = max_workers_gpu
    def pipeline(self, X, max_workers_cpu, n_tries=10, batch_size=400, seed=42) -> List[float]:
        results = []
        matrices_pairs = calculate_matrices_pairs(X, max_workers_cpu, n_tries, batch_size, seed, self.l)
        print("matrices done")
        print("available num of theads gpu " + str(self.max_workers_gpu))
        for part_gpu in range(int(len(matrices_pairs) // self.max_workers_gpu)):
            if self.max_workers_gpu * part_gpu >= len(matrices_pairs):
                break
            else:
                with ThreadPoolExecutor(
                    max_workers=min(
                        n_tries - self.max_workers_gpu * part_gpu, self.max_workers_gpu
                    )
                ) as executor_gpu:
                    futures = []
                    for i in range(executor_gpu._max_workers):
                        future = executor_gpu.submit(
                            self.__delta_hyp_GPU,
                            matrices_pairs[part_gpu * self.max_workers_gpu + i][0],
                            matrices_pairs[part_gpu * self.max_workers_gpu + i][1],
                        )
                        futures.append(future)
                    for j, future in enumerate(as_completed(futures)):
                        res = future.result()
                        print("res: " + str(res))
                        results.append(res)
        return results
    
    def __delta_hyp_GPU(self, dist_matrix, far_away_pairs):
        diam = np.max(dist_matrix)
        (
            n,
            x_coord_pairs,
            y_coord_pairs,
            adj_m,
            blockspergrid,
            threadsperblock,
            delta_res,
        ) = cuda_prep(far_away_pairs, dist_matrix, 32)
        print("pairs len " + str(len(far_away_pairs)))
        delta_hyp_CCL_GPU[blockspergrid, threadsperblock](
            n, x_coord_pairs, y_coord_pairs, adj_m, delta_res
        )
        del far_away_pairs
        del dist_matrix
        return 2 * delta_res[0] / diam, diam


class SeparateCartesianStrategy(PipelineStrategy):
    def __init__(self, l_multiple, mem_gpu_bound):
        self.l = l_multiple
        self.mem_gpu_bound = mem_gpu_bound
    def pipeline(self, X, max_workers_cpu, n_tries=10, batch_size=400, seed=42):
        results = []
        matrices_pairs = calculate_matrices_pairs(X, max_workers_cpu, n_tries, batch_size, seed, self.l)
        for i in range(len(matrices_pairs)):
            max_lines = calc_max_lines(self.mem_gpu_bound, len(matrices_pairs[i][1]))
            res_2 = self.__delta_cartesian_way_new(
                matrices_pairs[i][0],
                np.asarray(matrices_pairs[i][1], dtype=np.int32),
                max_lines,
            )
            results.append(res_2)

    def __delta_cartesian_way_new(self, X, far_away_pairs, batch_size):
        print("new way")
        diam = np.max(X)

        cartesian_size = int(len(far_away_pairs) * (len(far_away_pairs) - 1) / 2)

        batch_N = int(cartesian_size // batch_size) + 1
        deltas = np.empty(batch_N)

        print(f"shape X: {X.shape}")
        print(f"shape far_away_pairs: {far_away_pairs.shape}")
        print(f"all_size: {cartesian_size}")
        print(f"batch_size: {batch_size}")
        print(f"batches: {batch_N}")

        for i in range(batch_N):
            print(f"{i} batch started")
            (
                indicies
            ) = prepare_batch_indices_flat(
                far_away_pairs,
                i * batch_size,
                min((i + 1) * batch_size, cartesian_size),
                X.shape
            )

            batch = cython_batching.cython_flatten(indicies.ravel(), X.ravel()).reshape(-1, 6)

            print(batch.shape)
            print("batch built")

            (
                cartesian_dist_array,
                delta_res,
                threadsperblock,
                blockspergrid,
            ) = cuda_prep_cartesian(batch, 1024)
            delta_CCL_cartesian[blockspergrid, threadsperblock](
                cartesian_dist_array, delta_res
            )
            deltas[i] = delta_res[0]
            del cartesian_dist_array
        delta = max(deltas)
        return 2 * delta / diam, diam
