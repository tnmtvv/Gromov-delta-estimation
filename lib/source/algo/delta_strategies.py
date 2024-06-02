from abc import ABC, abstractmethod
import numpy as np
from hypdelta.delta import hypdelta
from hypdelta.calculus_utils import get_far_away_pairs

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from numba import typed, cuda


class DeltaStrategy(ABC):
    @abstractmethod
    def calculate_delta(self, X: np.ndarray) -> float:
        pass


class HeuristicTopKStrategy(DeltaStrategy):
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        const = min(50, dist_matrix.shape[0] - 1)
        # delta = delta_hyp_condensed_heuristic(dist_matrix, const)
        delta = hypdelta(dist_matrix, device="cpu", strategy="condensed")
        return 2 * delta / diam, diam


class CCLStrategy(DeltaStrategy):
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        delta = hypdelta(dist_matrix, device="cpu", l=0.05)
        return 2 * delta / diam, diam


# class CCLHeuristicStrategy(DeltaStrategy):
#     def calculate_delta(self, X: np.ndarray) -> float:
#         dist_matrix = pairwise_distances(X)
#         diam = np.max(dist_matrix)
#         far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 70)
#         max_iter = 100000
#         delta = delta_CCL_heuristic(dist_matrix, typed.List(far_away_pairs), max_iter)
#         return delta, diam


class CondensedStrategy(DeltaStrategy):
    def calculate_delta(self, X: np.ndarray) -> float:
        dist_matrix = pdist(X)
        diam = np.max(dist_matrix)
        delta = hypdelta(
            dist_matrix, device="cpu", strategy="condensed", heuristic="False"
        )
        return 2 * delta / diam, diam


class TrueDeltaGPUStrategy:
    def calculate_delta(self, X: np.ndarray) -> float:
        x = cuda.device_array(1)
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        delta = hypdelta(dist_matrix, device="gpu", strategy="naive")
        return 2 * delta / diam, diam


class TrueDeltaStrategy:
    def calculate_delta(self, X: np.ndarray) -> float:
        """
        Computes Gromov's delta-hyperbolicity value from distance matrix using the maxmin product.

        Parameters:
        -----------
        dismat : numpy.ndarray
            A square distance matrix of shape (n, n), where n is the number of nodes in a network.

        Returns:
        --------
        float
            The delta hyperbolicity value.

        Notes:
        ------
        This is a naive implementation that can be very inefficient on large datasets.
        Use `delta_hyp_condensed` function for scalable computations.
        """
        dist_matrix = pairwise_distances(X)
        diam = np.max(dist_matrix)
        delta = hypdelta(dist_matrix, device="cpu", strategy="naive")
        return 2 * delta / diam, diam
