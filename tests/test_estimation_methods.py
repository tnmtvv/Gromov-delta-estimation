import libcontext
from numba import typed
import pytest
from scipy.spatial.distance import pdist, squareform

from tests_utils import *
from lib.source.algo.CCL import *
from lib.source.algo.condenced import *
from lib.source.algo.true_delta import *


def test_CCL_true_delta():
    dist_matrix = generate_dists(500)
    far_away_pairs = get_far_away_pairs(dist_matrix, dist_matrix.shape[0] * 20)

    delta_CCL = delta_hyp_condensed_CCL(typed.List(far_away_pairs), dist_matrix)
    true_delta = delta_hyp(dist_matrix)

    assert delta_CCL == pytest.approx(true_delta, 0.1)


def test_condenced_true_delta():
    point_matr = generate_synthetic_points(500, 500)

    condenced_dist_matrix = pdist(point_matr)

    delta_condenced = delta_hyp_condensed(condenced_dist_matrix, point_matr.shape[0])
    delta_true = delta_hyp(squareform(condenced_dist_matrix))
    assert delta_true == pytest.approx(delta_true, 0.1)
