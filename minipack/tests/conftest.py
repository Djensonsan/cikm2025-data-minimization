import random
import torch.nn as nn
from recpack.algorithms.base import Algorithm
from scipy.sparse import csr_matrix
import torch
import numpy as np
import pytest
from minipack.logger_config import setup_logging

# This fixture will configure logging at the start of the test session, ensuring that all subsequent test executions use this configuration.
@pytest.fixture(scope="session", autouse=True)
def configure_logging(request):
    setup_logging(log_file_path='./minipack/tests/tests.log')

# Note: When running PyTests from Pycharm. Working directory matters for collecting conftest.py files.
# Note: Results will not be deterministic unless the "setup_seeds" fixture is used.
@pytest.fixture(autouse=True)
def setup_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.fixture(scope="function")
def small_random_target_matrix():
    # Sampling binary values
    Y_dense = np.random.randint(2, size=(20, 100))
    Y_sparse = csr_matrix(Y_dense)
    return Y_sparse


@pytest.fixture(scope="function")
def small_random_interaction_matrix():
    # Sampling binary values
    X_dense = np.random.randint(2, size=(20, 100))
    X_sparse = csr_matrix(X_dense)
    return X_sparse


@pytest.fixture(scope="function")
def small_random_test_matrix():
    # Sampling binary values
    X_dense = np.random.randint(2, size=(20, 100))
    X_sparse = csr_matrix(X_dense)
    return X_sparse

@pytest.fixture(scope="function")
def large_random_target_matrix():
    # Sampling binary values
    Y_dense = np.random.randint(2, size=(100, 300))
    Y_sparse = csr_matrix(Y_dense)
    return Y_sparse


@pytest.fixture(scope="function")
def large_random_interaction_matrix():
    # Sampling binary values
    X_dense = np.random.randint(2, size=(100, 300))
    X_sparse = csr_matrix(X_dense)
    return X_sparse


@pytest.fixture(scope="function")
def large_random_test_matrix():
    # Sampling binary values
    X_dense = np.random.randint(2, size=(100, 300))
    X_sparse = csr_matrix(X_dense)
    return X_sparse

@pytest.fixture(scope="function")
def target_matrices():
    Y_sparse_list = []
    # Generate 5 random CSR matrices
    for _ in range(2):
        # Sampling binary values
        Y_dense = np.random.randint(2, size=(100, 300))
        Y_sparse = csr_matrix(Y_dense)
        Y_sparse_list.append(Y_sparse)
    return Y_sparse_list

@pytest.fixture(scope="function")
def interaction_matrices():
    X_sparse_list = []
    # Generate 5 random CSR matrices
    for _ in range(2):
        # Sampling binary values
        X_dense = np.random.randint(2, size=(100, 300))
        X_sparse = csr_matrix(X_dense)
        X_sparse_list.append(X_sparse)
    return X_sparse_list

@pytest.fixture(scope="function")
def test_matrices():
    test_sparse_list = []
    # Generate 5 random CSR matrices
    for _ in range(2):
        # Sampling binary values
        X_dense = np.random.randint(2, size=(100, 300))
        X_sparse = csr_matrix(X_dense)
        test_sparse_list.append(X_sparse)
    return test_sparse_list

@pytest.fixture(scope="function")
def random_tensor():
    # Sampling binary values
    return torch.randint(high=2, size=(100, 300), dtype=torch.float)

class MockTorchModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MockModel(Algorithm):
    def __init__(self):
        self.model_ = MockTorchModel()

    def _check_fit_complete(self):
        pass

    def _fit(self, X: csr_matrix):
        pass

    def _predict(self, X: csr_matrix) -> csr_matrix:
        return self.model_(X)

class MockMetric:
    def __init__(self):
        pass

    def __call__(self, Y_true, Y_pred):
        return 0.5


@pytest.fixture(scope="function")
def mock_model():
    return MockModel()

def mock_metric():
    return MockMetric()

@pytest.fixture(scope="function")
def mock_model_torch():
    return MockTorchModel()

