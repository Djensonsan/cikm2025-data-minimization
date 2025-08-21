from scipy.sparse import csr_matrix

class BaseTargetEstimator:
    """Base class for target estimators."""

    @property
    def name(self):
        return self._class_name_and_attributes()

    def _class_name_and_attributes(self):
        """Dynamically generate class name and its attributes."""
        class_name = self.__class__.__name__
        attributes = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
        attributes_str = ', '.join(f"{key}={value}" for key, value in attributes.items())
        return f"{class_name}({attributes_str})"

    def get_params(self):
        """Retrieve the estimator's parameters."""
        params = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith('_')  # Avoid private attributes
        }
        return params

    def set_params(self, **params):
        """Set the estimator's parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}.")
        return self

    def estimate(self, Y_pred: csr_matrix) -> csr_matrix:
        """Estimate the ground truth."""
        raise NotImplementedError("Method not implemented")