import ray
import time
import logging
from typing import Union, List, Optional
from scipy.sparse import csr_matrix
from recpack.algorithms import Algorithm
from collections import Counter
from minipack.minimizers.result import MinimizationResult
from minipack.pipelines.registries import METRIC_REGISTRY
from minipack.pipelines.entries import MetricEntry
logger = logging.getLogger("minipack")


def log_ray_cluster_info(ray_info: dict):
    logger.info(" ---- Ray Cluster Information ----")
    logger.info(f" Ray Version: {ray.__version__}")

    dashboard_url = ray_info.address_info["webui_url"]
    logger.info(f" Dashboard URL: {dashboard_url}")

    logger.info(" **Available Resources**")
    cluster_resources = ray.cluster_resources()
    for resource_type, quantity in cluster_resources.items():
        logger.info(f"  {resource_type}: {quantity}")

    logger.info(" **Node Details**")
    for node in ray.nodes():
        logger.info(f"  - Node ID: {node['NodeID']}")
        logger.info(f"    Alive: {node['Alive']}")
        logger.info(f"    Resources: {node['Resources']}")
    logger.info(" ---- Ray Cluster Information ----")


def initialize_ray_cluster(
    num_cpus,
    object_store_memory,
    local_mode,
    include_dashboard,
    dashboard_host,
    ignore_reinit_error,
):
    if object_store_memory is not None:
        object_store_memory *= 1024**3  # Convert GB to bytes

    ray_info = ray.init(
        num_cpus=num_cpus,
        object_store_memory=object_store_memory,
        local_mode=local_mode,
        logging_level=logging.INFO,
        include_dashboard=include_dashboard,
        ignore_reinit_error=ignore_reinit_error,
        dashboard_host=dashboard_host,
    )
    log_ray_cluster_info(ray_info)
    return ray_info

def await_ray_tasks(task_references):
    results = []
    task_counter = Counter({"waiting": len(task_references), "completed": 0})
    start_wait_time = time.time()
    while task_references:
        ready, not_ready = ray.wait(task_references, num_returns=1)
        for task_reference in ready:
            result = ray.get(task_reference)
            results.append(result)

            # monitor progress
            task_counter["completed"] += 1
            task_counter["waiting"] -= 1

            # very crude ETA calculation
            wait_time = time.time() - start_wait_time
            eta = (
                wait_time * (task_counter["waiting"] / task_counter["completed"])
                if task_counter["completed"] > 0
                else 0
            )
            logger.info(
                 f"Tasks Waiting: {task_counter['waiting']}, Tasks Completed: {task_counter['completed']}, Wait Time (s): {wait_time:.2f}, ETA (s): {eta:.2f}"
            )
        task_references = not_ready
    return results

def compute_metrics(
        interaction_matrix: csr_matrix,
        test_matrix: csr_matrix,
        model: Algorithm,
        metric_entries: List[MetricEntry],
        remove_history: bool,
        result: MinimizationResult
):
    """Computes evaluation metrics for the minimized and full input matrices."""

    minimized_prediction_matrix = model.predict(result.minimized_matrix)
    prediction_matrix = model.predict(interaction_matrix)

    if remove_history:
        minimized_prediction_matrix -= minimized_prediction_matrix.multiply(interaction_matrix)
        prediction_matrix -= prediction_matrix.multiply(interaction_matrix)

    for metric_entry in metric_entries:
        metric_cls = METRIC_REGISTRY.get(metric_entry.name)
        if not metric_cls:
            raise ValueError(f"Metric '{metric_entry.name}' not found in registry.")

        metric = metric_cls(K=metric_entry.K) if metric_entry.K else metric_cls()

        # Calculate on full input
        metric.calculate(test_matrix, prediction_matrix)
        result.additional_batch_metrics[f"{metric.name}_test_full_input"] = metric.value
        result.additional_user_metrics[f"{metric.name}_test_full_input"] = metric.scores_.toarray().flatten()

        # Calculate on minimized input
        metric.calculate(test_matrix, minimized_prediction_matrix)
        result.additional_batch_metrics[f"{metric.name}_test_min_input"] = metric.value
        result.additional_user_metrics[f"{metric.name}_test_min_input"] = metric.scores_.toarray().flatten()

    return result