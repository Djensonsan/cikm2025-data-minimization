import os
import ray
import logging
import time
from tqdm.auto import tqdm
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Union, List
from recpack.algorithms import Algorithm
from minipack.target_estimators import BaseTargetEstimator
from minipack.pipelines.registries import MINIMIZATION_REGISTRY
from minipack.pipelines.entries import MinimizerEntry, MetricEntry
from minipack.logger_config import setup_logging
from minipack.pipelines.util import initialize_ray_cluster, await_ray_tasks, compute_metrics


logger = logging.getLogger("minipack")


# Note: Resource specifications can be used to limit the number of concurrent tasks.
@ray.remote(num_cpus=1)
def estimation_process(
    task_id: int,
    interaction_matrix: csr_matrix,
    test_matrix: csr_matrix,
    model: Algorithm,
    minimization_entry: MinimizerEntry,
    metric_entries: List[MetricEntry],
    target_estimator: BaseTargetEstimator,
    log_file_path: str,
    remove_history: bool = True,
):
    start_time = time.time()
    logger = setup_logging(log_file_path)
    logger.info(f"Started task with TID: {task_id} in process with PID: {os.getpid()}.")

    # Initialize minimizer
    minimizer_cls = MINIMIZATION_REGISTRY.get(minimization_entry.name)
    if minimizer_cls is None:
        raise ValueError(f"Minimization method '{minimization_entry.name}' not found in registry.")

    minimizer = minimizer_cls(**minimization_entry.params)
    minimizer.set_params(model=model)

    # Create target matrix using the estimator
    prediction_matrix = model._predict(interaction_matrix)
    if minimizer.remove_history:
        prediction_matrix -= prediction_matrix.multiply(interaction_matrix)
    target_matrix = target_estimator.estimate(prediction_matrix)

    result = minimizer.minimize(interaction_matrix, target_matrix)
    result.estimator_identifier = target_estimator.name

    # Calculate metrics on test data
    if test_matrix is not None:
        result = compute_metrics(
            interaction_matrix, test_matrix, model, metric_entries,
            remove_history, result
        )

    # Log completion time
    duration = time.time() - start_time
    logger.info(f"Completed task with TID: {task_id} in {duration:.2f} seconds.")

    return result

class Pipeline(object):
    def __init__(
        self,
        interaction_matrix: csr_matrix,
        test_matrix: Union[csr_matrix, None],
        model: Algorithm,
        minimization_entries: List[MinimizerEntry],
        metric_entries: List[MetricEntry],
        target_estimators: List[BaseTargetEstimator],
        save_results: bool,
        results_directory: str,
        log_file_path: str,
    ):
        self.interaction_matrix = interaction_matrix
        self.test_matrix = test_matrix
        self.model = model
        self.minimization_entries = minimization_entries
        self.metric_entries = metric_entries
        self.target_estimators = target_estimators
        self.save_results = save_results
        self.results_directory = results_directory
        self.log_file_path = log_file_path

        self.batch_results = None
        self.user_results = None

    def run(
        self,
        num_cpus=None,
        object_store_memory=None,
        local_mode: bool = False,
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        ignore_reinit_error=True,
    ):
        logger.info("Starting ray cluster.")
        ray_info = initialize_ray_cluster(
            num_cpus,
            object_store_memory,
            local_mode,
            include_dashboard,
            dashboard_host,
            ignore_reinit_error,
        )

        logger.info("Starting task submissions to ray cluster.")
        task_references = self._submit_ray_tasks()

        logger.info("Waiting for tasks to finish.")
        results = await_ray_tasks(task_references)

        logger.info("Shutting down Ray cluster.")
        ray.shutdown()

        logger.info("Aggregating and saving results.")
        self._process_results(results)

    def _submit_ray_tasks(self):
        # Use object refs for immutable shared objects
        model_ref = ray.put(self.model)

        # stores ray task references
        task_id = 0
        task_references = []
        for minimization_entry in tqdm(self.minimization_entries):
            for target_estimator in self.target_estimators:
                # Start a minimizer process for each batch
                task_reference = estimation_process.remote(
                    task_id,
                    self.interaction_matrix,
                    self.test_matrix,
                    model_ref,
                    minimization_entry,
                    self.metric_entries,
                    target_estimator,
                    self.log_file_path,
                    remove_history=True,
                )
                task_references.append(task_reference)
                task_id += 1

        return task_references

    @property
    def batch_results_file_path(self) -> str:
        return f"{self.results_directory}/batch_results.csv"

    @property
    def user_results_file_path(self) -> str:
        return f"{self.results_directory}/user_results.csv"

    def _process_results(self, minimization_results):
        self.batch_results = pd.concat([result.batch_results for result in minimization_results], ignore_index=True)
        self.user_results = pd.concat([result.user_results for result in minimization_results], ignore_index=True)
        if self.save_results:
            self._save_results()

    def _save_results(self):
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

        self.batch_results.to_csv(self.batch_results_file_path)
        self.user_results.to_csv(self.user_results_file_path)
