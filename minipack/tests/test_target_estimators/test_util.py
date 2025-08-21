import numpy as np
from minipack.target_estimators.util import get_top_K_ranks

def test_get_top_K1_ranks(Y_pred):
    K = 1
    result_matrix = get_top_K_ranks(Y_pred, K)
    result_matrix = result_matrix.toarray()
    expected_matrix = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])  # Correctly formatted 2D array
    np.testing.assert_array_equal(result_matrix, expected_matrix)

def test_get_top_K2_ranks(Y_pred):
    K = 2
    result_matrix = get_top_K_ranks(Y_pred, K)
    result_matrix = result_matrix.toarray()
    expected_matrix = np.array([[1, 0, 2, 0, 0], [0, 0, 0, 2, 1]])  # Correctly formatted 2D array
    np.testing.assert_array_equal(result_matrix, expected_matrix)
