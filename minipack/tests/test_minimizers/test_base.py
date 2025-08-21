import numpy as np
import pytest
import time
from unittest.mock import MagicMock
from scipy.sparse import csr_matrix
from recpack.algorithms.base import Algorithm
from recpack.metrics.base import Metric
from minipack.minimizers.base import BaseMinimizer, TimeoutManager

@pytest.fixture
def mock_metric():
    """Mock Metric object to speed up tests."""
    metric = MagicMock(spec=Metric)
    metric.calculate.return_value = csr_matrix(
        np.array([[0.8, 0.9, 1.0]])
    )  # Mocked output
    return metric


@pytest.fixture
def mock_model():
    """Mock Algorithm object to avoid real computation."""
    model = MagicMock(spec=Algorithm)
    model._predict.return_value = csr_matrix(
        np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    )  # Mocked predictions
    return model

@pytest.fixture
def sparse_matrix():
    """Returns a sample sparse matrix."""
    return csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]]))

@pytest.fixture
def minimizer(mock_metric, mock_model):
    """Minimizer with mocked dependencies."""
    return BaseMinimizer(
        metric=mock_metric, model=mock_model, eta=0.9, remove_history=True, timeout=5.0
    )

def test_minimizer_initialization(minimizer):
    """Test if minimizer initializes correctly."""
    assert minimizer.metric is not None
    assert minimizer.model is not None
    assert minimizer.eta == 0.9
    assert minimizer.remove_history is True
    assert minimizer.timeout == 5.0

@pytest.mark.parametrize("eta", [-0.5, "string", None])
def test_invalid_eta_values(eta, mock_metric, mock_model):
    """Test that invalid eta values raise errors."""
    with pytest.raises((ValueError, TypeError)):
        BaseMinimizer(metric=mock_metric, model=mock_model, eta=eta)


@pytest.mark.parametrize("timeout", [-1, "string", None])
def test_invalid_timeout_values(timeout, mock_metric, mock_model):
    """Test that invalid timeout values raise errors."""
    with pytest.raises((ValueError, TypeError)):
        BaseMinimizer(metric=mock_metric, model=mock_model, timeout=timeout)


@pytest.mark.parametrize("metric", [42, "invalid", object])
def test_invalid_metric_types(metric, mock_model):
    """Test that metric must be an instance of Metric."""
    with pytest.raises(TypeError, match="metric must be an instance of Metric"):
        BaseMinimizer(metric=metric, model=mock_model)


@pytest.mark.parametrize("model", [42, "invalid", object])
def test_invalid_model_types(model, mock_metric):
    """Test that model must be an instance of Algorithm."""
    with pytest.raises(TypeError, match="model must be an instance of Algorithm"):
        BaseMinimizer(metric=mock_metric, model=model)


def test_minimize_method_not_implemented(minimizer):
    """Ensure NotImplementedError is raised for _minimize()."""
    with pytest.raises(NotImplementedError):
        minimizer._minimize(csr_matrix((3, 3)), csr_matrix((3, 3)))


@pytest.mark.parametrize("invalid_matrix", [42, "invalid", np.array([[1, 2], [3, 4]])])
def test_invalid_interaction_matrix_in_minimize(minimizer, invalid_matrix):
    """Ensure minimize() raises TypeError for invalid interaction_matrix."""
    with pytest.raises(
        TypeError, match="interaction_matrix must be a scipy.sparse csr_matrix"
    ):
        minimizer.minimize(invalid_matrix, csr_matrix((3, 3)))


@pytest.mark.parametrize("invalid_matrix", [42, "invalid", np.array([[1, 2], [3, 4]])])
def test_invalid_target_matrix_in_minimize(minimizer, invalid_matrix):
    """Ensure minimize() raises TypeError for invalid target_matrix."""
    with pytest.raises(
        TypeError, match="target_matrix must be a scipy.sparse csr_matrix"
    ):
        minimizer.minimize(csr_matrix((3, 3)), invalid_matrix)


def test_minimize_without_metric():
    """Ensure minimize() raises ValueError if metric is not set."""
    minimizer = BaseMinimizer(model=MagicMock(spec=Algorithm))
    with pytest.raises(ValueError, match="Metric is not set"):
        minimizer.minimize(csr_matrix((3, 3)), csr_matrix((3, 3)))


def test_minimize_without_model():
    """Ensure minimize() raises ValueError if model is not set."""
    minimizer = BaseMinimizer(metric=MagicMock(spec=Metric))
    with pytest.raises(ValueError, match="Model is not set"):
        minimizer.minimize(csr_matrix((3, 3)), csr_matrix((3, 3)))


@pytest.mark.parametrize("invalid_matrix", [42, "invalid", np.array([[1, 2], [3, 4]])])
def test_invalid_matrices_in_forward(minimizer, invalid_matrix, sparse_matrix):
    """Ensure _forward() raises TypeError for invalid matrices."""
    with pytest.raises(TypeError):
        minimizer._forward(invalid_matrix, sparse_matrix, sparse_matrix)


def test_forward_without_metric(minimizer, sparse_matrix):
    """Ensure _forward() raises ValueError if metric is not set."""
    minimizer.metric = None
    with pytest.raises(ValueError, match="Metric is not set"):
        minimizer._forward(sparse_matrix, sparse_matrix, sparse_matrix)


def test_forward_without_model(minimizer, sparse_matrix):
    """Ensure _forward() raises ValueError if model is not set."""
    minimizer.model = None
    with pytest.raises(ValueError, match="Model is not set"):
        minimizer._forward(sparse_matrix, sparse_matrix, sparse_matrix)


def test_timeout_handling(minimizer):
    """Ensure timeout event triggers correctly."""
    minimizer.result_builder = MagicMock()

    minimizer.timeout_manager._timeout_callback()

    assert minimizer.timeout_manager.has_timed_out()

def test_minimizer_forward(minimizer, sparse_matrix):
    """Test _forward method using mocks."""
    scores = minimizer._forward(sparse_matrix, sparse_matrix, sparse_matrix)

    # Ensure the method is calling the mocked methods
    minimizer.model._predict.assert_called_once()
    minimizer.metric.calculate.assert_called_once()

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)  # Ensure it returns a flattened array


@pytest.mark.parametrize("invalid_matrix", [42, "invalid", np.array([[1, 2], [3, 4]])])
def test_invalid_matrices_in_set_thresholds(minimizer, invalid_matrix, sparse_matrix):
    """Ensure _set_thresholds() raises TypeError for invalid matrices."""
    with pytest.raises(TypeError):
        minimizer._set_thresholds(invalid_matrix, sparse_matrix)


def test_set_thresholds_without_metric(minimizer, sparse_matrix):
    """Ensure _set_thresholds() raises ValueError if metric is not set."""
    minimizer.metric = None
    with pytest.raises(ValueError, match="Metric not set"):
        minimizer._set_thresholds(sparse_matrix, sparse_matrix)


def test_set_thresholds_without_model(minimizer, sparse_matrix):
    """Ensure _set_thresholds() raises ValueError if model is not set."""
    minimizer.model = None
    with pytest.raises(ValueError, match="Model not set"):
        minimizer._set_thresholds(sparse_matrix, sparse_matrix)


@pytest.mark.parametrize("invalid_eta", [-1, None])
def test_set_thresholds_invalid_eta(minimizer, sparse_matrix, invalid_eta):
    """Ensure _set_thresholds() raises ValueError for invalid eta."""
    minimizer.eta = invalid_eta
    with pytest.raises((ValueError, TypeError)):
        minimizer._set_thresholds(sparse_matrix, sparse_matrix)


@pytest.mark.parametrize("eta", [0.0, 0.9, 1.0])
def test_set_thresholds(minimizer, sparse_matrix, eta):
    """Test threshold calculation for different eta values."""
    minimizer.eta = eta
    thresholds = minimizer._set_thresholds(sparse_matrix, sparse_matrix)

    assert isinstance(thresholds, np.ndarray)
    assert thresholds.shape == (3,)
    assert np.all(
        thresholds == eta * minimizer.metric.calculate.return_value.toarray().flatten()
    )

def test_timeout_manager_start_and_timeout():
    timeout_manager = TimeoutManager(timeout=0.1)
    timeout_manager.start()

    # Wait for the timeout to occur
    time.sleep(0.2)

    assert timeout_manager.has_timed_out() is True

def test_timeout_manager_cancel():
    timeout_manager = TimeoutManager(timeout=0.1)
    timeout_manager.start()
    timeout_manager.cancel()

    # Wait to ensure the timer would have expired
    time.sleep(0.2)

    assert timeout_manager.has_timed_out() is False

def test_timeout_manager_lock():
    timeout_manager = TimeoutManager(timeout=0.1)
    timeout_manager.lock()

    # Attempt to start the timer
    timeout_manager.start()

    # Wait to ensure the timer would have expired
    time.sleep(0.2)

    assert timeout_manager.has_timed_out() is False

    # Unlock and start the timer
    timeout_manager.unlock()
    timeout_manager.start()

    # Wait for the timeout to occur
    time.sleep(0.2)

    assert timeout_manager.has_timed_out() is True

def test_timeout_manager_cancel_when_locked():
    timeout_manager = TimeoutManager(timeout=0.1)
    timeout_manager.start()
    timeout_manager.lock()

    # Attempt to cancel the timer
    timeout_manager.cancel()

    # Wait to ensure the timer would have expired
    time.sleep(0.2)

    assert timeout_manager.has_timed_out() is True

