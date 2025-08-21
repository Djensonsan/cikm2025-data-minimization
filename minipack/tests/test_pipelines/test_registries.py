from minipack.pipelines.registries import MINIMIZATION_REGISTRY, METRIC_REGISTRY

def test_minimization_registry():
    assert "RandomSelectionMinimizer" in MINIMIZATION_REGISTRY
    assert "GreedyForwardMinimizer" in MINIMIZATION_REGISTRY
    assert "GreedyBeamForwardMinimizer" in MINIMIZATION_REGISTRY

def test_metric_registry():
    assert "CalibratedRecallK" in METRIC_REGISTRY
    assert "HitK" in METRIC_REGISTRY
    assert "NDCGK" in METRIC_REGISTRY