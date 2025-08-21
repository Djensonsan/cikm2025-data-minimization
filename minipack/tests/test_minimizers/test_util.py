import pytest
import numpy as np
from scipy.sparse import lil_matrix
from minipack.minimizers.util import sort_lil_matrix, repeat_lil_vector, deduplicate_lil


def test_sort_lil_matrix_inplace():
    X1 = lil_matrix((3, 5))
    X1[0, [3, 1, 4]] = [10, 5, 7]
    X1[1, [2, 0]] = [3, 8]
    X1[2, [4, 1]] = [4, 6]

    sort_lil_matrix(X1, inplace=True)

    assert X1.rows[0] == [1, 4, 3]
    assert X1.data[0] == [5, 7, 10]
    assert X1.rows[1] == [2, 0]
    assert X1.data[1] == [3, 8]
    assert X1.rows[2] == [4, 1]
    assert X1.data[2] == [4, 6]

def test_sort_lil_matrix_return_new():
    X1 = lil_matrix((3, 5))
    X1[0, [3, 1, 4]] = [10, 5, 7]
    X1[1, [2, 0]] = [3, 8]
    X1[2, [4, 1]] = [4, 6]

    X2 = sort_lil_matrix(X1, inplace=False)

    assert X2.rows[0] == [1, 4, 3]
    assert X2.data[0] == [5, 7, 10]
    assert X2.rows[1] == [2, 0]
    assert X2.data[1] == [3, 8]
    assert X2.rows[2] == [4, 1]
    assert X2.data[2] == [4, 6]

    # Ensure the original matrix is unchanged
    # Note: LIL sorts column indices.
    assert X1.rows[0] == [1, 3, 4]
    assert X1.data[0] == [5, 10, 7]
    assert X1.rows[1] == [0, 2]
    assert X1.data[1] == [8, 3]
    assert X1.rows[2] == [1, 4]
    assert X1.data[2] == [6, 4]

def test_csr_conversion_equivalence():
    # Create a sample LIL matrix
    X1 = lil_matrix((3, 5))
    X1[0, [3, 1, 4]] = [10, 5, 7]
    X1[1, [2, 0]] = [3, 8]
    X1[2, [4, 1]] = [4, 6]

    # Convert to CSR without sorting
    csr_before_sort = X1.tocsr()

    # Sort the LIL matrix
    sort_lil_matrix(X1, inplace=True)

    # Convert to CSR after sorting
    csr_after_sort = X1.tocsr()

    # Check that the CSR matrices are identical by comparing dense representations
    np.testing.assert_array_equal(csr_before_sort.toarray(), csr_after_sort.toarray())

def test_csr_conversion_empty_matrix():
    # Create an empty LIL matrix
    X1 = lil_matrix((3, 5))

    # Convert to CSR without sorting
    csr_before_sort = X1.tocsr()

    # Sort the LIL matrix
    sort_lil_matrix(X1, inplace=True)

    # Convert to CSR after sorting
    csr_after_sort = X1.tocsr()

    # Check that the CSR matrices are identical by comparing dense representations
    np.testing.assert_array_equal(csr_before_sort.toarray(), csr_after_sort.toarray())

def test_empty_matrix():
    X1 = lil_matrix((3, 5))
    sort_lil_matrix(X1, inplace=True)

    for i in range(X1.shape[0]):
        assert X1.rows[i] == []
        assert X1.data[i] == []

def test_single_row_matrix():
    X1 = lil_matrix((1, 5))
    X1[0, [3, 1, 4]] = [10, 5, 7]

    sort_lil_matrix(X1, inplace=True)

    assert X1.rows[0] == [1, 4, 3]
    assert X1.data[0] == [5, 7, 10]

def test_unsorted_indices():
    X1 = lil_matrix((3, 5))
    X1[0, [4, 1, 3]] = [7, 5, 10]
    X1[1, [0, 2]] = [8, 3]

    sort_lil_matrix(X1, inplace=True)

    assert X1.rows[0] == [1, 4, 3]
    assert X1.data[0] == [5, 7, 10]
    assert X1.rows[1] == [2, 0]
    assert X1.data[1] == [3, 8]

def test_repeat_lil_vector_basic():
    X = lil_matrix([[1, 2, 3]])
    times = 3
    expected = lil_matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    result = repeat_lil_vector(X, times)
    assert result.shape == (times, X.shape[1])
    assert (result.toarray() == expected.toarray()).all()

def test_repeat_lil_vector_exception():
    X = lil_matrix([[1, 2, 3], [4, 5, 6]])
    times = 3
    with pytest.raises(ValueError):
        repeat_lil_vector(X, times)

def test_repeat_lil_vector_zero_times():
    X = lil_matrix([[1, 2, 3]])
    times = 0
    expected = lil_matrix((0, 3))
    result = repeat_lil_vector(X, times)
    assert result.shape == (times, X.shape[1])
    assert (result.toarray() == expected.toarray()).all()

def test_deduplicate_lil_basic():
    X = lil_matrix([[1, 0, 2], [1, 0, 2], [0, 3, 1]])
    expected = lil_matrix([[1, 0, 2], [0, 3, 1]])
    result = deduplicate_lil(X)
    assert result.shape[0] == 2
    # Note: This test assumes deduplication does not preserve order. Adjust as necessary.
    assert {(tuple(row), tuple(data)) for row, data in zip(result.rows, result.data)} == \
           {(tuple(row), tuple(data)) for row, data in zip(expected.rows, expected.data)}

def test_deduplicate_lil_empty():
    X = lil_matrix((0, 3))
    result = deduplicate_lil(X)
    assert result.shape == (0, 3)

def test_deduplicate_lil_non_lil_input():
    X = [[1, 2, 3], [1, 2, 3]]
    with pytest.raises(ValueError):
        deduplicate_lil(X)

def test_repeat_lil_vector_negative_times():
    X = lil_matrix([[1, 2, 3]])
    times = -1
    with pytest.raises(ValueError):
        repeat_lil_vector(X, times)

def test_deduplicate_lil_all_unique_rows():
    X = lil_matrix([[1, 0], [0, 1]])
    result = deduplicate_lil(X)
    assert result.shape == X.shape
    assert (result.toarray() == X.toarray()).all()

def test_deduplicate_lil_nonempty_matrix():
    X = lil_matrix([[1, 0, 0], [1, 0, 0], [0, 1, 1]])
    expected = lil_matrix([[1, 0, 0], [0, 1, 1]])  # Assuming deduplication does not preserve order
    result = deduplicate_lil(X)
    # Due to the unordered nature of sets, we convert to sets of tuples for comparison
    assert {(tuple(row), tuple(data)) for row, data in zip(result.rows, result.data)} == \
           {(tuple(row), tuple(data)) for row, data in zip(expected.rows, expected.data)}

def test_deduplicate_lil_empty_matrix():
    X = lil_matrix((0, 0))
    result = deduplicate_lil(X)
    assert result.shape == (0, 0)