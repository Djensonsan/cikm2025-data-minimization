import numpy as np
from typing import List, Union
from scipy.sparse import lil_matrix, csr_matrix

def pad_with_nan(array, target_length):
    """
    Pads a 1D numpy array with NaN values to the specified target length.

    Parameters:
        array (np.ndarray): The input 1D numpy array.
        target_length (int): The desired length of the output array.

    Returns:
        np.ndarray: A new array of length `target_length`, padded with NaN values if needed.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if not array.ndim == 1:
        raise ValueError("Input array must be 1D.")
    if not isinstance(target_length, int) or target_length < len(array):
        raise ValueError("Target length must be an integer greater than or equal to the length of the array.")

    # Create a new array filled with NaN values
    padded_array = np.full(target_length, np.nan)

    # Copy the original array into the new array
    padded_array[:len(array)] = array

    return padded_array

def sort_lil_matrix(matrix, inplace=True, reverse=False):
    """
    Sorts the rows of a Scipy sparse LIL matrix based on the values in each row.

    Args:
        matrix (lil_matrix): The LIL sparse matrix to sort.
        inplace (bool): If True, sort the matrix in place. If False, return a new sorted matrix.
        reverse (bool): If True, sort in descending order. If False, sort in ascending order.

    Returns:
        lil_matrix: A new sorted LIL matrix if inplace=False; otherwise, None.
    """
    if not inplace:
        matrix = matrix.copy()

    for row_idx in range(matrix.shape[0]):
        # Extract column indices and values for the current row
        row_indices = matrix.rows[row_idx]
        row_values = matrix.data[row_idx]

        # Pair column indices with values and sort by values
        paired_data = list(zip(row_indices, row_values))

        # Sort by values, with the order determined by the `reverse` parameter
        paired_data.sort(key=lambda x: x[1], reverse=reverse)

        # Unzip the sorted pairs back into indices and values
        if paired_data:
            matrix.rows[row_idx], matrix.data[row_idx] = map(list, zip(*paired_data))
        else:
            matrix.rows[row_idx], matrix.data[row_idx] = [], []

    if not inplace:
        return matrix

def repeat_lil_vector(vector: lil_matrix, times: int) -> lil_matrix:
    """Repeats a LIL matrix with a single row a specified number of times.

    This function takes a LIL matrix, expected to contain only a single row, and creates a new LIL matrix
    where the original row is repeated 'times' number of times. This is useful for algorithms that require
    the same data to be duplicated across multiple rows.

    Args:
        vector (lil_matrix): The LIL matrix to be repeated, expected to contain only a single row.
        times (int): The number of times to repeat the row.

    Returns:
        lil_matrix: A new LIL matrix where the single row from X is repeated 'times' number of times.
    """
    if vector.shape[0] != 1:
        raise ValueError("Input matrix X is expected to contain only a single row.")

    # Creating a new LIL matrix with the appropriate dimensions
    X_repeated = lil_matrix((times, vector.shape[1]))
    for i in range(times):
        X_repeated.rows[i] = vector.rows[0].copy()
        X_repeated.data[i] = vector.data[0].copy()
    return X_repeated

def deduplicate_lil(matrix):
    """Remove duplicate rows from a LIL (List of Lists) matrix.

    This function takes a LIL matrix and returns a new LIL matrix with duplicate rows removed.
    It identifies unique rows based on non-zero indices and their corresponding values.
    Note that this function does not preserve the original row order of the matrix.

    Args:
        matrix (lil_matrix): The input sparse matrix to be deduplicated.

    Returns:
        lil_matrix: A new LIL matrix with duplicate rows removed.

    Raises:
        ValueError: If the input matrix is not in LIL format.

    Note:
        The function checks if the input matrix is empty and, if so, returns it directly.
    """
    # Ensure the matrix is in LIL format
    if not isinstance(matrix, lil_matrix):
        raise ValueError("Input matrix must be a LIL matrix.")

    # Check if the matrix is empty
    if matrix.shape[0] == 0:
        return matrix

    # Convert the rows to a set of tuples for uniqueness
    unique_rows = set()
    for i in range(matrix.shape[0]):
        row_data = matrix.rows[i]
        row_values = matrix.data[i]
        # Ensure the row is sorted by index for consistent comparison
        sorted_row = sorted(zip(row_data, row_values))
        unique_rows.add(tuple(sorted_row))

    # Create a new LIL matrix for the unique rows
    unique_matrix = lil_matrix((len(unique_rows), matrix.shape[1]))

    # Fill the unique matrix with data from unique rows
    for i, row in enumerate(unique_rows):
        indices, data = zip(*row) if row else ([], [])
        unique_matrix.rows[i] = list(indices)
        unique_matrix.data[i] = list(data)

    return unique_matrix

def row_slice(
    rows_to_slice: Union[np.ndarray, List[int]], *args: Union[np.ndarray, csr_matrix]
) -> List[Union[np.ndarray, csr_matrix]]:
    """
    Slices the provided matrices or arrays based on the specified rows.

    Args:
        rows_to_slice (Union[np.ndarray, List[int]]): Indices of rows to be updated.
        *args (Union[np.ndarray, csr_matrix]): Sliceable objects such as numpy arrays
            or scipy sparse CSR matrices.

    Returns:
        List[Union[np.ndarray, csr_matrix]]: List of sliced objects, in the same order
            as provided in *args.
    """
    return [obj[rows_to_slice] for obj in args]
