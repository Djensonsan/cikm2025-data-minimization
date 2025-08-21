import numpy as np
from minipack.metrics import NDCG
from minipack.target_estimators import ExponentialDecayEstimator


def test_NDCG_simple_binary_no_cache(Y_true_binary_simplified, Y_pred):
    metric = NDCG(K=2)
    metric.calculate(Y_true_binary_simplified, Y_pred)

    IDCG = sum(1 / np.log2((i + 1) + 1) for i in range(0, 1))

    expected_value = ((1 / np.log2(2 + 1)) / IDCG + 1) / 2

    np.testing.assert_almost_equal(metric.value, expected_value)


def test_NDCG_binary_no_cache(Y_true_binary, Y_pred):
    metric = NDCG(K=2)
    metric.calculate(Y_true_binary, Y_pred)

    # user 0 has 2 correct items, user 2 has three correct items
    # however, K is 2 so ideal for user 2 is IDCG2
    IDCG = 1 + 1 / np.log2(3)

    expected_value = ((1 + (1 / np.log2(3))) / IDCG + (0 + (1 / np.log2(3))) / IDCG) / 2

    np.testing.assert_almost_equal(metric.value, expected_value)


def test_NDCG_parametric_with_cache_1(Y_pred, Y_pred_switched):
    """
    Checks expected value of NDCG on model predictions as ground truth.
    These predictions are non-binary indicating the predicted relevance for each item (representing probabilities).
    """
    estimator = ExponentialDecayEstimator(K=3, gamma=1)

    Y_pred = estimator.estimate(Y_pred)

    metric = NDCG(K=3, ground_truth_template=estimator)

    scores = metric.calculate(Y_pred, Y_pred_switched)
    scores = scores.toarray().flatten()
    # For user 0: has three spot-on item predictions:
    # IDCG = 1.0 / log(2) + 0.5 /log(3) + 0.0 / log(4)
    # DCG = 1.0 / log(2) + 0.5 /log(3) + 0.0 / log(4)
    # NDCG = DCG / IDCG = 1.0
    np.testing.assert_almost_equal(scores[0], 1.0)
    # For user 2: has a correct item ranking at position 1, but positions 2 and 3 have been switched
    # IDCG = 1.0 / log(2) + 0.5 / log(3) + 0.0 / log(4)
    # DCG = 1.0 / log(2) + 0.0 /log(3) + 0.5 / log(4)
    # NDCG = DCG / IDCG = 0.95023442
    np.testing.assert_almost_equal(scores[1], 0.95023442)


def test_NDCG_parametric_with_cache_2(Y_pred, Y_pred_switched):
    """
    Checks expected value of NDCG on model predictions as ground truth.
    These predictions are non-binary indicating the predicted relevance for each item (representing probabilities).
    """
    estimator = ExponentialDecayEstimator(K=3, gamma=0.1)

    Y_pred = estimator.estimate(Y_pred)

    metric = NDCG(K=3, ground_truth_template=estimator)

    scores = metric.calculate(Y_pred, Y_pred_switched)
    scores = scores.toarray().flatten()
    # For user 0: has three spot-on item predictions:
    # IDCG = 1.0 / log(2) + 0.42 /log(3) + 0.0 / log(4)
    # DCG = 1.0 / log(2) + 0.42 /log(3) + 0.0 / log(4)
    # NDCG = DCG / IDCG = 1.0
    np.testing.assert_almost_equal(scores[0], 1.0)
    # For user 2: has a correct item ranking at position 1, but positions 2 and 3 have been switched
    # IDCG = 1.0 / log(2) + 0.42 /log(3) + 0.0 / log(4) =
    # DCG = 1.0 / log(2) + 0.0 /log(3) + 0.42 / log(4) = 1.21
    # NDCG = DCG / IDCG = 0.95624571
    np.testing.assert_almost_equal(scores[1], 0.95624571)


def test_NDCG_parametric_no_cache_1(Y_pred, Y_pred_switched):
    """
    Checks expected value of NDCG on model predictions as ground truth.
    These predictions are non-binary indicating the predicted relevance for each item (representing probabilities).
    """
    estimator = ExponentialDecayEstimator(K=3, gamma=1)

    Y_pred = estimator.estimate(Y_pred)

    metric = NDCG(K=3)

    scores = metric.calculate(Y_pred, Y_pred_switched)
    scores = scores.toarray().flatten()
    # For user 0: has three spot-on item predictions:
    # IDCG = 1.0 / log(2) + 0.5 /log(3) + 0.0 / log(4)
    # DCG = 1.0 / log(2) + 0.5 /log(3) + 0.0 / log(4)
    # NDCG = DCG / IDCG = 1.0
    np.testing.assert_almost_equal(scores[0], 1.0)
    # For user 2: has a correct item ranking at position 1, but positions 2 and 3 have been switched
    # IDCG = 1.0 / log(2) + 0.5 / log(3) + 0.0 / log(4)
    # DCG = 1.0 / log(2) + 0.0 /log(3) + 0.5 / log(4)
    # NDCG = DCG / IDCG = 0.95023442
    np.testing.assert_almost_equal(scores[1], 0.95023442)


def test_NDCG_parametric_no_cache_2(Y_pred, Y_pred_switched):
    """
    Checks expected value of NDCG on model predictions as ground truth.
    These predictions are non-binary indicating the predicted relevance for each item (representing probabilities).
    """
    estimator = ExponentialDecayEstimator(K=3, gamma=0.1)

    Y_pred = estimator.estimate(Y_pred)

    metric = NDCG(K=3)

    scores = metric.calculate(Y_pred, Y_pred_switched)
    scores = scores.toarray().flatten()
    # For user 0: has three spot-on item predictions:
    # IDCG = 1.0 / log(2) + 0.42 /log(3) + 0.0 / log(4)
    # DCG = 1.0 / log(2) + 0.42 /log(3) + 0.0 / log(4)
    # NDCG = DCG / IDCG = 1.0
    np.testing.assert_almost_equal(scores[0], 1.0)
    # For user 2: has a correct item ranking at position 1, but positions 2 and 3 have been switched
    # IDCG = 1.0 / log(2) + 0.42 /log(3) + 0.0 / log(4) =
    # DCG = 1.0 / log(2) + 0.0 /log(3) + 0.42 / log(4) = 1.21
    # NDCG = DCG / IDCG = 0.95624571
    np.testing.assert_almost_equal(scores[1], 0.95624571)