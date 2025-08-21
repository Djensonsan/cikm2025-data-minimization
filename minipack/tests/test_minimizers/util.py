from scipy.sparse import csr_matrix

def assert_common_results(result, interaction_matrix, target_matrix, expect_timeout=False):
    """Helper function to assert common properties of minimization results."""
    assert result.batch_number_of_users == interaction_matrix.shape[0]
    assert result.batch_number_of_users_processed == interaction_matrix.shape[0]
    assert result.batch_percentage_of_users_processed == 1.0
    assert result.batch_original_number_of_input_interactions == interaction_matrix.nnz
    assert result.batch_number_of_target_interactions == target_matrix.nnz
    assert result.batch_runtime > 0
    assert result.timeout_occurred == expect_timeout
    assert result.batch_minimized_number_of_input_interactions < result.batch_original_number_of_input_interactions
    assert result.batch_minimization_ratio < 1.0
    assert result.batch_sample_count > 0
    assert is_subset_csr(result.minimized_matrix, interaction_matrix)

def is_subset_csr(A, B):
    """
    Check if CSR matrix A is a subset of CSR matrix B.
    A is a subset of B if all non-zero elements in A are also non-zero and equal in B.

    Args:
        A (csr_matrix): The smaller matrix to check as a subset.
        B (csr_matrix): The larger matrix to compare against.

    Returns:
        bool: True if A is a subset of B, False otherwise.
    """
    # Check non-zero values of A in B
    A_coo = A.tocoo()  # Convert to COO for easier iteration
    B_coo = B.tocoo()

    # Create dictionaries of non-zero elements for comparison
    A_dict = {(i, j): v for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data)}
    B_dict = {(i, j): v for i, j, v in zip(B_coo.row, B_coo.col, B_coo.data)}

    # Verify all non-zero elements of A are in B and have the same value
    for key, value in A_dict.items():
        if key not in B_dict or B_dict[key] != value:
            return False

    return True