from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List
from lib.source.algo.delta_strategies import *
from decimal import *
from timeit import default_timer as timer
from hypdelta.fast import delta_CCL_gpu
from hypdelta.cartesian import delta_cartesian


from lib.source.algo.algo_utils import (
    calculate_matrices_pairs,
)


class PipelineStrategy(ABC):
    @abstractmethod
    def pipeline(self, **kwargs) -> List[float]:
        pass


class UniteStrategy(PipelineStrategy):
    def __init__(self, strategy: DeltaStrategy):
        self.strategy = strategy

    def pipeline(
        self, X, max_workers_cpu, n_tries=10, batch_size=400, seed=42
    ) -> List[float]:
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
                            batch_idx = rng.permutation(n_objects)
                        else:
                            batch_idx = rng.choice(
                                n_objects, batch_size, replace=False, shuffle=True
                            )
                        item_space = X[batch_idx]
                        print("unite strategy")
                        print("batch done")
                        print("item space shape" + str(item_space.shape))
                        future = executor.submit(
                            self.strategy.calculate_delta, X=item_space
                        )
                        futures.append(future)
                    for i, future in enumerate(as_completed(futures)):
                        delta_rel, diam = res = future.result()
                        print("res: " + str(res))
                        results.append(res)
        return results


class SeparateStrategy:
    def __init__(self, l_multiple, max_workers_gpu):
        self.l = l_multiple
        self.max_workers_gpu = max_workers_gpu

    def pipeline(
        self, X, max_workers_cpu, n_tries=10, batch_size=400, seed=42
    ) -> List[float]:
        results = []
        matrices_pairs = calculate_matrices_pairs(
            X, max_workers_cpu, n_tries, batch_size, seed, self.l
        )
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
                            delta_CCL_gpu,
                            matrices_pairs[part_gpu * self.max_workers_gpu + i][0],
                            far_away_pairs=matrices_pairs[
                                part_gpu * self.max_workers_gpu + i
                            ][1],
                        )
                        futures.append(future)
                    for j, future in enumerate(as_completed(futures)):
                        dist_matrix = matrices_pairs[part_gpu * self.max_workers_gpu + i][0]
                        res = future.result(), np.max(dist_matrix)
                        print("res: " + str(res))
                        results.append(res)
        return results


class SeparateCartesianStrategy(PipelineStrategy):
    def __init__(self, l_multiple, mem_gpu_bound):
        self.l = l_multiple
        self.mem_gpu_bound = mem_gpu_bound

    def pipeline(self, X, max_workers_cpu, n_tries=10, batch_size=400, seed=42):
        results = []
        matrices_pairs = calculate_matrices_pairs(
            X, max_workers_cpu, n_tries, batch_size, seed, self.l
        )
        for i in range(len(matrices_pairs)):
            res_2 = delta_cartesian(
                matrices_pairs[i][0],
                far_away_pairs=np.asarray(matrices_pairs[i][1], dtype=np.int32),
                all_threads=1024,
                mem_gpu_bound=self.mem_gpu_bound,
            ), np.max(matrices_pairs[i][0])
            results.append(res_2)
        return results
