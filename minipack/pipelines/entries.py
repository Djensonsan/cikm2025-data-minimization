from recpack.pipelines.registries import AlgorithmEntry, OptimisationMetricEntry, MetricEntry
from typing import Dict, NamedTuple, List, Optional, Any

class MinimizerEntry(NamedTuple):
    """
    Config class to represent a minimizer when configuring the pipeline.

    Args:
        name (str): Name of the algorithm.
        params (Optional[Dict[str, Any]]): Parameters that do not require optimization as key-value pairs,
            where the key is the name of the hyperparameter and the value is the value it should take.
    """

    name: str
    params: Optional[Dict[str, Any]] = None