import os
import pickle
import torch

# Note: RecPack does not have functionality to save and load models. Should be added to RecPack Algorithm class. For now, we use a MixIn.
class SerializableModel:
    """
    A mixin that provides save and load functionality for models.
    """
    @property
    def filename(self):
        """Name of the file where the model will be saved."""
        return f"{self.name}.pkl"

    def save(self, results_directory=None):
        """
        Saves the model to the specified directory or the current working directory if none is specified.

        Args:
            results_directory (str, optional): Directory to save the model file. Created if it doesn't exist. Defaults to the current working directory.
        """
        if results_directory is None:
            results_directory = os.getcwd()

        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        model_file_path = os.path.join(results_directory, self.filename)

        with open(model_file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_file_path):
        """
        Loads a model from a file.

        Args:
            model_file_path (str): Path to the model file.

        Returns:
            An instance of the class.
        """
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)

        if not isinstance(model, cls):
            raise ValueError(f"Loaded model is not an instance of {cls.__name__}")

        return model

# Note: RecPack does have functionality to save and load Torch models. However, the filename is according to a weird convention and you can't set a results path.
class SerializableTorchModel:
        """
        Mixin to override bad serialization logic in base classes for PyTorch models.
        Provides methods to save and load models and manage filenames.
        """
        @property
        def filename(self):
            """Name of the file where the model will be saved."""
            return f"{self.name}.pth"

        def save(self, results_directory=None):
            """
            Save the Torch model to a file.

            Args:
                results_directory (str, optional): Directory to save the file.
                                                   Defaults to the current working directory.
            """
            if results_directory is None:
                results_directory = os.getcwd()

            if not os.path.exists(results_directory):
                os.makedirs(results_directory)

            file_path = os.path.join(results_directory, self.filename)
            torch.save(self.model_, file_path)

        def load(self, model_file_path):
            """
            Load a Torch model from a file.

            Args:
                model_file_path (str): Path to the file.
            """
            with open(model_file_path, "rb") as f:
                self.model_ = torch.load(f, weights_only=False)