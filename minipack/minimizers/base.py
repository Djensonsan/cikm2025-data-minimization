import threading
import logging
import numpy as np
from recpack.algorithms.base import Algorithm
from recpack.metrics.base import Metric

from minipack.minimizers.result import (
    MinimizationResultBuilder,
    MinimizationResult,
)
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

logger = logging.getLogger("minipack")

class TimeoutManager:
    def __init__(self, timeout):
        self.timeout = timeout
        self._timeout_event = threading.Event()
        self.timer = threading.Timer(self.timeout, self._timeout_callback)
        self._locked = False

    def start(self):
        if not self._locked:
            self.timer.start()
            self._timer_started = True

    def cancel(self):
        if not self._locked:
            self.timer.cancel()
            self._timeout_event.clear()  # Reset the event for potential reuse

    def _timeout_callback(self):
        self._timeout_event.set()

    def has_timed_out(self):
        return self._timeout_event.is_set()

    def lock(self):
        self._locked = True

    def unlock(self):
        self._locked = False

class BaseMinimizer(BaseEstimator):
    def __init__(
        self,
        metric: Metric = None,
        model: Algorithm = None,
        eta: float = 0.9,
        remove_history: bool = True,
        timeout: float = float("inf"),
        timeout_manager: TimeoutManager = None,
    ):
        """Initialize the BaseMinimizer.

        Args:
            metric (Metric, optional): A function that computes the metric to be used for minimization. This could be any callable
                that takes as input the predictions and true labels and returns floats representing the metric values.
                Defaults to None.
            model (Algorithm, optional): The algorithm instance adhering to the RecPack Algorithm interface,
                used here for making inferences during the minimization process. Defaults to None.

        Note:
            This is a base class and is intended to be subclassed. The actual minimization logic should be implemented
            in the subclasses.
        """
        super().__init__()

        # Note: We allow eta > 1.0. This corresponds to removing data to improve predictive performance.
        if not isinstance(eta, (int, float)) or not 0 <= eta:
            raise ValueError(f"eta must be a float between 0 and 1, got {eta}")

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"timeout must be a positive number, got {timeout}")

        if metric is not None and not isinstance(metric, Metric):
            raise TypeError(f"metric must be an instance of Metric, got {type(metric)}")

        if model is not None and not isinstance(model, Algorithm):
            raise TypeError(
                f"model must be an instance of Algorithm, got {type(model)}"
            )
        self.metric = metric
        self.model = model
        self.eta = eta
        self.remove_history = remove_history
        self.timeout = timeout
        if timeout_manager is not None:
            self.timeout_manager = timeout_manager
        else:
            self.timeout_manager = TimeoutManager(timeout)

    @property
    def name(self):
        """Name of the object's class."""
        return self.__class__.__name__

    @property
    def identifier(self):
        """Name of the object, constructed by combining the class name with its parameters.

        The name is created by recreating the initialization call with the current parameters of the object.
        This provides a concise representation of the object with its configuration.

        Example:
            If the object is an instance of a class named 'Algorithm' and was initialized with
            parameters 'param_1=value', then the identifier would be "Algorithm(param_1=value)".

        Returns:
            str: A string representing the name of the object, combining the class name and its parameters.
        """

        def value_repr(value):
            """Represents the value in the desired format, checking for 'identifier' or 'name' properties."""
            if hasattr(value, "identifier"):
                return value.identifier
            elif hasattr(value, "name"):
                return value.name
            else:
                return value

        paramstring = ",".join(
            f"{k}={value_repr(v)}" for k, v in self.get_params(deep=False).items()
        )
        return f"{self.name}({paramstring})"

    def __str__(self):
        return self.name

    def set_params(self, **params):
        """Set the parameters of the minimizer (estimator).

        This method allows for setting or updating the parameters of the estimator object.
        It accepts variable keyword arguments representing the parameters and their values.

        Args:
            params: Variable keyword arguments representing the estimator parameters.
                    Each key-value pair in 'params' updates the corresponding attribute of the object.

        Example:
            To update the 'alpha' parameter of an estimator object, call `set_params(alpha=0.5)`.
        """
        super().set_params(**params)

    def minimize(
        self, interaction_matrix: csr_matrix, target_matrix: csr_matrix
    ) -> MinimizationResult:
        """Performs minimization on the given dataset.

        Args:
            interaction_matrix (csr_matrix): A sparse matrix representing input data, where rows correspond to users and columns to items.
            target_matrix (csr_matrix): A sparse matrix representing ground truth data, structured similarly to "interaction_matrix".

        Returns:
            MinimizationResult: An object containing the results of the minimization process.

        Raises:
            ValueError: If `metric`, `model`, or `performance_threshold` is not set before the method is called.

        Note:
            This method is intended to be a high-level orchestrator for minimization. The actual optimization logic
            should be implemented in the `_minimize` method within subclasses.
        """
        if not isinstance(interaction_matrix, csr_matrix):
            raise TypeError(
                f"interaction_matrix must be a scipy.sparse csr_matrix, got {type(interaction_matrix)}"
            )

        if not isinstance(target_matrix, csr_matrix):
            raise TypeError(
                f"target_matrix must be a scipy.sparse csr_matrix, got {type(target_matrix)}"
            )

        if self.metric is None:
            raise ValueError(
                "Metric is not set. Please set the metric before running the minimizer."
            )
        if not isinstance(self.metric, Metric):
            raise TypeError(
                f"metric must be an instance of Metric, got {type(self.metric)}"
            )

        if self.model is None:
            raise ValueError(
                "Model is not set. Please set the model before running the minimizer."
            )
        if not isinstance(self.model, Algorithm):
            raise TypeError(
                f"model must be an instance of Algorithm, got {type(self.model)}"
            )

        if not hasattr(self, "thresholds") or self.thresholds is None:
            self._set_thresholds(interaction_matrix, target_matrix)

        self.result_builder = MinimizationResultBuilder()
        self.result_builder.set_minimizer_identifier(self.identifier)
        self.result_builder.set_input_statistics(interaction_matrix)
        self.result_builder.set_target_statistics(target_matrix)
        self.result_builder.set_per_user_performance_threshold(self.thresholds)

        self.timeout_manager.start()
        try:
            self.result_builder = self._minimize(interaction_matrix, target_matrix)
        finally:
            self.result_builder.set_timeout_occurred(self.timeout_manager.has_timed_out())
            self.timeout_manager.cancel()
        return self.result_builder.build()

    def _minimize(
        self, interaction_matrix: csr_matrix, target_matrix: csr_matrix
    ) -> MinimizationResultBuilder:
        """Performs minimization on the given dataset, to be implemented by subclasses.

        Args:
            interaction_matrix (csr_matrix): A sparse matrix representing input data, where rows correspond to users and columns to items.
            target_matrix (csr_matrix): A sparse matrix representing ground truth data, structured similarly to "interaction_matrix".

        Returns:
            MinimizationResult: An object containing the results of the minimization process.

        Raises:
            NotImplementedError: This exception is raised to indicate that the method must be implemented in subclasses,
                                 as it is intended to be an abstract method in this base class.
        """
        raise NotImplementedError("Method not implemented")

    def _forward(
        self,
        input_matrix: csr_matrix,
        history_matrix: csr_matrix,
        target_matrix: csr_matrix,
    ) -> np.ndarray:
        """
        Performs a forward computation:
            1. Creates predictions using the input matrix and the model.
            2. (Optional) Filters the predictions using the history matrix.
            3. Scores the predictions using the metric and target_matrix.

        The function assumes a valid metric and model have been initialized. If they
        are not set, a `ValueError` is raised. The output is a flattened NumPy array
        of calculated scores.

         Args:
            input_matrix (csr_matrix): A sparse matrix representing input data, where rows correspond to users and columns to items.
            history_matrix (csr_matrix): A sparse matrix representing the full user history, structured similarly to "input_matrix".
            target_matrix (csr_matrix): A sparse matrix representing ground truth data, structured similarly to "input_matrix".

        Returns:
            Scores: An numpy array containing the scores for each user's predictions.
        """
        for matrix, name in zip([input_matrix, history_matrix, target_matrix],
                                ["input_matrix", "history_matrix", "target_matrix"]):
            if not isinstance(matrix, csr_matrix):
                raise TypeError(f"{name} must be a scipy.sparse csr_matrix, got {type(matrix)}")

        if self.metric is None:
            raise ValueError("Metric is not set.")
        if not isinstance(self.metric, Metric):
            raise TypeError(f"metric must be an instance of Metric, got {type(self.metric)}")

        if self.model is None:
            raise ValueError("Model is not set.")
        if not isinstance(self.model, Algorithm):
            raise TypeError(f"model must be an instance of Algorithm, got {type(self.model)}")

        prediction_matrix = self.model._predict(input_matrix)

        if self.remove_history:
            prediction_matrix = prediction_matrix - prediction_matrix.multiply(
                history_matrix
            )

        scores = self.metric.calculate(target_matrix, prediction_matrix)

        return scores.toarray().flatten()

    def _set_thresholds(
        self, interaction_matrix: csr_matrix, target_matrix: csr_matrix
    ) -> np.ndarray:
        """Sets the performance threshold for the minimization process.

        The threshold is determined as a percentage (defined by `eta`) of the performance metric obtained using the entire dataset.
        Optionally supports returning the threshold as a PyTorch tensor for minimizers that require PyTorch.

        Args:
            interaction_matrix (csr_matrix): A sparse matrix representing input data, where rows correspond to users and columns to items.
            target_matrix (csr_matrix): A sparse matrix representing ground truth data, structured similarly to "interaction_matrix".

        Returns:
            np.ndarray: The calculated performance threshold as a numpy array. The returned value indicates the minimally acceptable performance of the model
                when using a minimized dataset.

        Raises:
            ValueError: If either the metric or the model has not been set prior to calling this method.
        """
        if not isinstance(interaction_matrix, csr_matrix):
            raise TypeError(f"interaction_matrix must be a scipy.sparse csr_matrix, got {type(interaction_matrix)}")

        if not isinstance(target_matrix, csr_matrix):
            raise TypeError(f"target_matrix must be a scipy.sparse csr_matrix, got {type(target_matrix)}")

        if self.metric is None:
            raise ValueError("Metric not set. Please set the metric before setting a performance threshold.")
        if not isinstance(self.metric, Metric):
            raise TypeError(f"metric must be an instance of Metric, got {type(self.metric)}")

        if self.model is None:
            raise ValueError("Model not set. Please set the model before setting a performance threshold.")
        if not isinstance(self.model, Algorithm):
            raise TypeError(f"model must be an instance of Algorithm, got {type(self.model)}")

        if self.eta is None:
            raise ValueError("Eta not set. Please set the eta before setting a performance threshold.")
        if not isinstance(self.eta, (int, float)) or not 0 <= self.eta:
            raise TypeError(f"eta must be a float between 0 and 1, got {self.eta}")

        scores = self._forward(
            input_matrix=interaction_matrix,
            history_matrix=interaction_matrix,
            target_matrix=target_matrix,
        )

        self.thresholds = self.eta * scores

        return self.thresholds

    def save(self, path: str):
        """Saves the minimizer's current state and/or results to a file.

        This method is intended to serialize the minimizer's attributes, including its internal state and any results
        it has produced, to a file specified by the `path` parameter. The exact format of the saved data (e.g., pickle,
        JSON, CSV) depends on the implementation in subclasses. Implementers should ensure that all necessary information
        for later restoration or analysis of the minimizer's state and results is included.

        Args:
            path (str): The path to the file where the minimizer's state and/or results will be saved. This should include
                        the file name and extension appropriate to the format being used.

        Raises:
            NotImplementedError: Indicates that the method must be implemented in subclasses.
        """
        raise NotImplementedError("Method not implemented")
