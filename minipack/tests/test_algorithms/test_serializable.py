import os
import torch
import tempfile
from minipack.tests.test_algorithms.mocks import MockTorchModel

def test_serializable_model(mock_model):
    with tempfile.TemporaryDirectory() as results_directory:
        model_file_path = os.path.join(results_directory, mock_model.filename)

        mock_model.save(results_directory)
        assert os.path.exists(model_file_path), "File was not saved."

        loaded_instance = mock_model.__class__.load(model_file_path)

        assert loaded_instance.name == mock_model.name
        assert loaded_instance.value == mock_model.value

def test_serializable_torch_model(mock_torch_model):
    with tempfile.TemporaryDirectory() as results_directory:
        model_file_path = os.path.join(results_directory, mock_torch_model.filename)

        mock_torch_model.save(results_directory)
        assert os.path.exists(model_file_path), "File was not saved."

        loaded_instance = MockTorchModel()
        loaded_instance.load(model_file_path)

        assert isinstance(loaded_instance.model_, torch.nn.Module)
        assert isinstance(loaded_instance.model_.linear, torch.nn.Linear)