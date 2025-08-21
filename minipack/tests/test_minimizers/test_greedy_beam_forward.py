import pytest
import numpy as np
from minipack.metrics import NDCG
from minipack.algorithms.nearest_neighbour import ItemKNN
from minipack.minimizers import GreedyBeamForwardMinimizer
from minipack.tests.test_minimizers.util import assert_common_results

def test_initialization_with_custom_parameters():
    """Ensure that a custom beam depth is correctly set."""
    mini = GreedyBeamForwardMinimizer(beam_depth=5)
    assert mini.beam_depth == 5


def test_invalid_beam_depth_raises_exception():
    """Ensure an invalid beam depth raises an exception."""
    with pytest.raises(ValueError):
        GreedyBeamForwardMinimizer(beam_depth=-1)


def test_minimize_with_defaults(small_random_interaction_matrix, small_random_target_matrix):
    """Test minimization with default settings on a randomized model."""
    model = ItemKNN()
    model.fit(small_random_interaction_matrix)

    minimizer = GreedyBeamForwardMinimizer(metric=NDCG(K=2), model=model, remove_history=False)
    result = minimizer.minimize(small_random_interaction_matrix, small_random_target_matrix)

    assert_common_results(result, small_random_interaction_matrix, small_random_target_matrix)
    assert result.batch_constraint_satisfaction == 1.0

def test_minimize_timeout(small_random_interaction_matrix, small_random_target_matrix):
    """Test timeout behavior in minimization."""
    model = ItemKNN()
    model.fit(small_random_interaction_matrix)

    minimizer = GreedyBeamForwardMinimizer(metric=NDCG(K=2), model=model, timeout=1e-6)
    result = minimizer.minimize(small_random_interaction_matrix, small_random_target_matrix)

    assert result.batch_constraint_satisfaction == 1.0
    assert result.batch_percentage_of_users_processed < 1.0
    assert result.timeout_occurred == True
    assert np.isnan(result.per_user_minimization_ratios).any()
    assert np.isnan(result.per_user_scores).any()
    assert np.isnan(result.per_user_runtimes).any()
    assert np.isnan(result.per_user_sample_counts).any()


@pytest.mark.parametrize(
    "eta, expected_sample_count, expected_minimized_interactions",
    [
        (0.0, 3, 0),  # eta=0.0 should remove all interactions
        (1.0, 7, 3),  # eta=1.0 should remove a single interaction
        (0.9, 7, 3),  # eta=0.9 should remove a single interaction
    ],
)
def test_minimize_with_different_eta(
    eta, expected_sample_count, expected_minimized_interactions,
    simple_interaction_matrix, simple_target_matrix, simple_train_matrix
):
    """
    Assert that the algorithm finds the smallest input history with the expected sample count.

    Our input history (simple_interaction_matrix) is:
    [0.00, 1.00, 0.00]
    [1.00, 0.00, 0.00]
    [1.00, 1.00, 0.00]

    The similarity matrix (S) of ItemKNN is:
    [0.00, 0.50, 0.81]
    [0.50, 0.00, 0.81]
    [0.81, 0.81, 0.00]

    Our ground truth (simple_target_matrix) is:
    [0.00, 0.00, 1.00]
    [0.00, 0.00, 1.00]
    [0.00, 0.00, 1.00]

    For User 0: No item can be dropped since there is only one in the input history.
    For User 1: No item can be dropped since there is only one in the input history.
    For User 2: Since items 0, 1 of the input history have the same similarity with item 2. We can drop one item and still achieve the same NDCG.
    """
    model = ItemKNN(K=3)
    model.fit(simple_train_matrix)

    minimizer = GreedyBeamForwardMinimizer(metric=NDCG(K=2), model=model, eta=eta)
    result = minimizer.minimize(simple_interaction_matrix, simple_target_matrix)

    assert_common_results(result, simple_interaction_matrix, simple_target_matrix)
    assert result.batch_sample_count == expected_sample_count
    assert result.batch_minimized_number_of_input_interactions == expected_minimized_interactions
    assert result.batch_constraint_satisfaction == 1.0
