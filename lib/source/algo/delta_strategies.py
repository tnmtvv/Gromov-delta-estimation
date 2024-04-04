from abc import ABC, abstractmethod
import numpy as np
from lib.source.algo.CCL import (
    delta_CCL_heuristic,
    delta_hyp_CCL_GPU,
    delta_hyp_condensed_CCL,
)
from lib.source.algo.condenced import delta_hyp_condensed, delta_hyp_condensed_heuristic
from lib.source.algo.true_delta import true_delta_gpu
from lib.source.algo.algo_utils import (
    get_far_away_pairs,
    cuda_prep_true_delta,
    cuda_prep,
)

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from numba import typed


class DeltaStrategy(ABC):
    @abstractmethod
    def calculate_delta(self, X: np.ndarray) -> float:
        pass


class HeuristicTopKStrategy(DeltaStrategy):
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        const = min(50, X.shape[0] - 1)
        mode = "top_k"
        delta = delta_hyp_condensed_heuristic(
            dist_matrix, dist_matrix.shape[0], const, mode
        )
        return delta, diam


class HeuristicTopRandStrategy(DeltaStrategy):
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        const = min(50, X.shape[0] - 1)
        mode = "top_rand"
        delta = delta_hyp_condensed_heuristic(
            dist_matrix, dist_matrix.shape[0], const, mode
        )
        return delta, diam


class CCLStrategy(DeltaStrategy):
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 50)
        delta = delta_hyp_condensed_CCL(typed.List(far_away_pairs), dist_matrix)
        return delta, diam


class GPUStrategy(DeltaStrategy):
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 50)
        (
            n,
            x_coord_pairs,
            y_coord_pairs,
            adj_m,
            blockspergrid,
            threadsperblock,
            delta_res,
        ) = cuda_prep(far_away_pairs, dist_matrix, 32)
        delta_hyp_CCL_GPU[blockspergrid, threadsperblock](
            n, x_coord_pairs, y_coord_pairs, adj_m, delta_res
        )
        delta, _ = 2 * delta_res[0] / np.max(dist_matrix)
        return delta, diam


class CCLHeuristicStrategy(DeltaStrategy):
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 50)
        max_iter = 100000
        delta = delta_CCL_heuristic(dist_matrix, typed.List(far_away_pairs), max_iter)
        return delta, diam


class CondencedStrategy(DeltaStrategy):
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pdist(X)
        diam = np.max(dist_matrix)
        delta = delta_hyp_condensed(dist_matrix, X.shape[0])
        return delta, diam


class TrueDeltaGPUStrategy:
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        adj_m, k, delta_res, threadsperblock, blockspergrid = cuda_prep_true_delta(
            dist_matrix
        )
        true_delta_gpu[blockspergrid, threadsperblock](adj_m, delta_res, k)
        delta = delta_res[0]
        return delta, diam
