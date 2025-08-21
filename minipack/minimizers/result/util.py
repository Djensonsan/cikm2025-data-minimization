def nonempty_rows(X):
    """
    Return the number of nonempty rows in a scipy CSR matrix.

    Parameters:
        X (csr_matrix): A scipy.sparse.csr_matrix object.

    Returns:
        int: Number of nonempty rows.
    """
    return (X.getnnz(axis=1) > 0).sum()


def percentage_of_nonempty_rows(X):
    """
    Calculate the percentage of rows in a scipy CSR matrix that contain at least one nonzero element.

    Parameters:
    X (csr_matrix): A scipy.sparse.csr_matrix object.

    Returns:
    float: Percentage of rows with at least one nonzero value.
    """
    # Calculate the number of non-zero elements in each row
    nonzero_per_row = X.getnnz(axis=1)

    # Count rows with at least one nonzero element
    rows_with_nonzero = (nonzero_per_row > 0).sum()

    # Calculate the percentage of rows with at least one nonzero element
    percentage = (rows_with_nonzero / X.shape[0])

    return percentage