import libcontext
from numba import typed
import pytest
from scipy.spatial.distance import pdist, squareform

from tests_utils import *
from lib.source.algo.CCL import *
from lib.source.algo.condenced import *
from lib.source.algo.true_delta import *

from lib.source.algo.delta_strategies import *

from lib.source.algo.delta import delta_hyp_GPU


def test_CCL_true_delta():
    dist_matrix = generate_dists(500)

    strategy_CCL = CCLStrategy()
    strategy_true = TrueDeltaGPUStrategy()

    delta_CCL = strategy_CCL.calculate_delta(dist_matrix)
    delta_true = strategy_true.calculate_delta(dist_matrix)

    assert delta_CCL == pytest.approx(delta_true, 0.001)


def test_CCL_GPU():
    dist_matrix = generate_dists(500)

    strategy_CCL = CCLStrategy()
    strategy_gpu = GPUStrategy()

    delta_GPU = strategy_gpu.calculate_delta(dist_matrix)
    delta_CCL = strategy_CCL.calculate_delta(dist_matrix)

    assert delta_GPU == pytest.approx(delta_CCL, 0.001)


def test_GPU_true_delta():
    dist_matrix = generate_dists(500)

    strategy_CCL = GPUStrategy()
    strategy_true = TrueDeltaGPUStrategy()

    delta_GPU = strategy_CCL.calculate_delta(dist_matrix)
    delta_true = strategy_true.calculate_delta(dist_matrix)

    assert delta_GPU == pytest.approx(delta_true, 0.001)


def test_condenced_true_delta():
    dist_matrix = generate_dists(500)

    strategy_condenced = CondencedStrategy()
    strategy_true = TrueDeltaGPUStrategy()

    delta_condenced = strategy_condenced.calculate_delta(dist_matrix)
    delta_true = strategy_true.calculate_delta(dist_matrix)

    assert delta_condenced == pytest.approx(delta_true, 0.001)
