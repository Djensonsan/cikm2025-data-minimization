import pytest
from minipack.tests.test_algorithms.mocks import MockModel, MockTorchModel

@pytest.fixture
def mock_model():
    return MockModel(name="test", value=42)

@pytest.fixture
def mock_torch_model():
    return MockTorchModel()