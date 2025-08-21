import numpy as np
from scipy.sparse import csr_matrix
from minipack.target_estimators import ExponentialDecayEstimator
from minipack.target_estimators.util import get_top_K_ranks

def test_exponential_decay_function():
    estimator = ExponentialDecayEstimator(K=3, gamma=1)
    x = np.array([1, 2, 3])
    expected_output = np.array([1, 0.5, 0])
    output = estimator._estimator_function(x)
    np.testing.assert_almost_equal(output, expected_output)

    estimator = ExponentialDecayEstimator(K=3, gamma=0.5)
    x = np.array([1, 2, 3] )
    expected_output = np.array([1, 0.45741788, 0])
    output = estimator._estimator_function(x)
    np.testing.assert_almost_equal(output, expected_output)

def test_exponential_decay_estimator_initialization():
    estimator = ExponentialDecayEstimator(K=5, gamma=0.5)
    assert estimator.K == 5
    assert estimator.gamma == 0.5

def test_estimate_functionality():
    # Create a sample sparse matrix
    data = [5, 4, 3, 2, 1]
    row_indices = [0, 0, 0, 0, 0]
    col_indices = [0, 1, 2, 3, 4]
    Y = csr_matrix((data, (row_indices, col_indices)), shape=(1, 5))

    estimator = ExponentialDecayEstimator(K=3, gamma=0.5)
    result = estimator.estimate(Y)

    # Check if the result is still a csr_matrix
    assert isinstance(result, csr_matrix)

    # Check if the data has been transformed correctly
    top_k_ranks = get_top_K_ranks(Y, estimator.K)
    expected_data = estimator._estimator_function(top_k_ranks.data)
    assert np.allclose(result.data, expected_data)

def test_estimate_with_empty_matrix():
    Y = csr_matrix((5, 5))
    estimator = ExponentialDecayEstimator(K=3, gamma=0.5)
    result = estimator.estimate(Y)

    # The result should also be an empty matrix
    assert result.nnz == 0

def test_estimate_with_large_values():
    data = [500, 400, 300, 200, 100]
    row_indices = [0, 0, 0, 0, 0]
    col_indices = [0, 1, 2, 3, 4]
    Y = csr_matrix((data, (row_indices, col_indices)), shape=(1, 5))

    estimator = ExponentialDecayEstimator(K=3, gamma=2.0)
    result = estimator.estimate(Y)

    # Check if the result is still a csr_matrix
    assert isinstance(result, csr_matrix)

    # Check if the data has been transformed correctly
    top_k_ranks = get_top_K_ranks(Y, estimator.K)
    expected_data = estimator._estimator_function(top_k_ranks.data)
    assert np.allclose(result.data, expected_data)

def test_estimate_with_negative_gamma():
    data = [5, 4, 3, 2, 1]
    row_indices = [0, 0, 0, 0, 0]
    col_indices = [0, 1, 2, 3, 4]
    Y = csr_matrix((data, (row_indices, col_indices)), shape=(1, 5))

    estimator = ExponentialDecayEstimator(K=3, gamma=-1.0)
    result = estimator.estimate(Y)

    # Check if the result is still a csr_matrix
    assert isinstance(result, csr_matrix)

    # Check if the data has been transformed correctly
    top_k_ranks = get_top_K_ranks(Y, estimator.K)
    expected_data = estimator._estimator_function(top_k_ranks.data)
    assert np.allclose(result.data, expected_data)

