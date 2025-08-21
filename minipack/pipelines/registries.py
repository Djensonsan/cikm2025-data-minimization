from recpack.pipelines.registries import Registry
from minipack import metrics
from minipack import algorithms
from minipack import minimizers
from minipack import scenarios
from minipack import target_estimators

class AlgorithmRegistry(Registry):
    """Registry for easy retrieval of algorithm types by name.

    The registry comes preregistered with wrappers of recpack algorithms.
    """

    def __init__(self):
        super().__init__(algorithms)

class MetricRegistry(Registry):
    """Registry for easy retrieval of metric types by name.

    The registry comes preregistered with all the recpack metrics.
    """

    def __init__(self):
        super().__init__(metrics)

class MinimizationRegistry(Registry):
    """Registry for easy retrieval of minimization algorithm types by name.

    This is the full registry, which contains both single algorithms and the hybrid minimization class.

    The registry comes preregistered with all implemented minimization algorithms.
    """

    def __init__(self):
        super().__init__(minimizers)

class ScenarioRegistry(Registry):
    """Registry for easy retrieval of scenario types by name.

    The registry comes preregistered with all implemented scenarios.
    """

    def __init__(self):
        super().__init__(scenarios)

class TargetEstimatorRegistry(Registry):
    """Registry for easy retrieval of target estimator types by name.

    The registry comes preregistered with all implemented target estimators.
    """

    def __init__(self):
        super().__init__(target_estimators)

ALGORITHM_REGISTRY = AlgorithmRegistry()
METRIC_REGISTRY = MetricRegistry()
MINIMIZATION_REGISTRY = MinimizationRegistry()
SCENARIO_REGISTRY = ScenarioRegistry()
TARGET_ESTIMATOR_REGISTRY = TargetEstimatorRegistry()