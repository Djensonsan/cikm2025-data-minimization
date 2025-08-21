import logging
import torch.nn as nn
from itertools import product
from collections.abc import Iterable
from scipy.sparse import csr_matrix
from typing import Tuple, Union, Dict, List, Optional, Any
from recpack.algorithms import Algorithm
from minipack.pipelines.registries import MINIMIZATION_REGISTRY, METRIC_REGISTRY, TARGET_ESTIMATOR_REGISTRY
from minipack.pipelines.entries import MinimizerEntry, MetricEntry
from minipack.pipelines.target_estimation.pipeline import Pipeline

logger = logging.getLogger('minipack')


class PipelineBuilder(object):
    def __init__(self):
        self.metric_entries = {}
        self.minimizer_entries = []

    def _arg_to_str(self, arg: Union[type, str]) -> str:
        if type(arg) == type:
            arg = arg.__name__

        elif type(arg) != str:
            raise TypeError(f"Argument should be string or type, not {type(arg)}")

        return arg

    def _init_minimizers(self, minimizer: Dict[str, Any], params: Optional[Dict[str, Any]] = None):
        minimizer = self._arg_to_str(minimizer)

        if minimizer not in MINIMIZATION_REGISTRY:
            raise ValueError(
                f"Minimization algorithm {minimizer} could not be resolved."
            )

        if params is not None:
            if "metric"  in params:
                metric = params["metric"]
                if metric["name"] not in METRIC_REGISTRY:
                    raise ValueError(f"Metric {metric} could not be resolved.")

                metric_cls = METRIC_REGISTRY.get(metric["name"])
                metric = metric_cls(**metric["params"])

                params["metric"] = metric

        # Recursive initialization of minimizers
        if minimizer == "HybridMinimizer":
            minimizers = []
            for sub_minimizer in params["minimizer_entries"]:
                sub_minimizer = self._init_minimizers(sub_minimizer['name'], sub_minimizer['params'])
                minimizers.append(sub_minimizer)
            params["minimizer_entries"] = minimizers
        return MinimizerEntry(minimizer, params or {})

    def add_minimizer(
        self,
        minimizer: Union[str, type],
        params: Optional[Dict[str, Any]] = None,
    ):
        minimizer_entry = self._init_minimizers(minimizer, params)
        self.minimizer_entries.append(minimizer_entry)


    def add_target_estimator(self, estimator: Union[str, type], grid: Dict[str, List] = None):
        estimator = self._arg_to_str(estimator)

        if estimator not in TARGET_ESTIMATOR_REGISTRY:
            raise ValueError(f"Target estimator {estimator} could not be resolved.")

        target_estimator_cls = TARGET_ESTIMATOR_REGISTRY.get(estimator)

        # Generate all combinations of the grid parameters
        param_combinations = list(product(*grid.values()))

        # Create a list of dictionaries with parameter combinations
        param_dicts = [dict(zip(grid.keys(), params)) for params in param_combinations]

        self.estimators = [target_estimator_cls(**params) for params in param_dicts]

    def add_metric(
        self,
        metric: Union[str, type],
        K: Optional[Union[List, int]] = None,
    ):
        # Make it so it's possible to add metrics by their class as well.
        metric = self._arg_to_str(metric)

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric} could not be resolved.")

        if isinstance(K, Iterable):
            for k in K:
                self.add_metric(metric, k)
        elif K is not None:
            # TODO Should we validate these K values to see if they make sense?
            # Check if metric already exists
            metric_name = f"{metric}_{K}"

            if metric_name in self.metric_entries:
                logger.warning(f"Metric {metric_name} already exists.")
            else:
                self.metric_entries[metric_name] = MetricEntry(metric, K)
        else:
            # Bit of a hack to pass none, but it's the best I can do I think.
            self.metric_entries[metric] = MetricEntry(metric, K)

    @property
    def interaction_matrix(self):
        return self._interaction_matrix

    @interaction_matrix.setter
    def interaction_matrix(self, interaction_matrix: csr_matrix):
        if  type(interaction_matrix) != csr_matrix:
            raise TypeError(f"Argument should be a CSR Matrix, not {type(interaction_matrix)}")
        self._interaction_matrix = interaction_matrix

    @property
    def test_matrix(self):
        return self._test_matrix

    @test_matrix.setter
    def test_matrix(self, test_matrix: csr_matrix):
        if type(test_matrix) != csr_matrix:
            raise TypeError(
                f"Argument should be a CSR Matrix, not {type(test_matrix)}"
            )
        self._test_matrix = test_matrix

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Algorithm):
        if not isinstance(model, Algorithm):
            raise RuntimeError("Model must be RecPack algorithm.")

        self._model = model

        # Check whether the model has been fit
        self._model._check_fit_complete()

    @property
    def save_results(self):
        return self._save_results

    @save_results.setter
    def save_results(self, value):
        if not isinstance(value, bool):
            raise TypeError(f"Save results should be a boolean, not {type(value)}")
        self._save_results = value

    @property
    def results_directory(self):
        return self._results_directory

    @results_directory.setter
    def results_directory(self, results_directory: str):
        if not isinstance(results_directory, str):
            raise TypeError(
                f"Results directory should be a string, not {type(results_directory)}"
            )
        self._results_directory = results_directory

    @property
    def log_file_path(self):
        return self._log_file_path

    @log_file_path.setter
    def log_file_path(self, log_file_path: str):
        if not isinstance(log_file_path, str):
            raise TypeError(
                f"Log file path should be a string, not {type(log_file_path)}"
            )
        self._log_file_path = log_file_path

    def _check_readiness(self):
        if not hasattr(self, "interaction_matrix"):
            raise RuntimeError("No input data available, can't construct pipeline")

        if not hasattr(self, "model"):
            raise RuntimeError(
                "No recommendation model specified, can't construct pipeline"
            )

        if len(self.minimizer_entries) == 0:
            raise RuntimeError(
                "No minimization algorithms specified, can't construct pipeline"
            )

        if not hasattr(self, "estimators"):
            raise RuntimeError(
                "No ground truth estimators specified, can't construct pipeline"
            )

        # Check whether the model is differentiable when using GradientOptimizer
        if any(
            entry.name == "GradientMinimizer" for entry in self.minimizer_entries
        ):
            if not hasattr(self.model, "model_"):
                raise RuntimeError(
                    "Need algorithm with .model_ parameter, can't construct pipeline"
                )
            if not issubclass(type(self.model.model_), nn.Module):
                raise RuntimeError(
                    "Model parameter (.model_) must be of class nn.Module when using GradientOptimizer, can't construct pipeline"
                )

        if not hasattr(self, "results_directory"):
            raise RuntimeError(
                "No results directory specified, can't construct pipeline"
            )

        if not hasattr(self, "log_file_path"):
            raise RuntimeError(
                "No log file path specified, can't construct pipeline"
            )

    def build(self) -> Pipeline:
        self._check_readiness()

        return Pipeline(
            self.interaction_matrix,
            self.test_matrix if hasattr(self, "test_matrix") else None,
            self.model,
            self.minimizer_entries,
            list(self.metric_entries.values()),
            self.estimators,
            self.save_results if hasattr(self, "save_results") else False,
            self.results_directory,
            self.log_file_path,
        )
