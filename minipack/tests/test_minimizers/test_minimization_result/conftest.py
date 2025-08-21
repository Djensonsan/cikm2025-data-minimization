import pytest
from scipy.sparse import csr_matrix
from minipack.minimizers.result.result_builder import MinimizationResultBuilder

@pytest.fixture
def builder():
    return MinimizationResultBuilder()

@pytest.fixture
def X():
    return csr_matrix([[1, 0, 0], [0, 1, 1], [1, 1, 0]])

