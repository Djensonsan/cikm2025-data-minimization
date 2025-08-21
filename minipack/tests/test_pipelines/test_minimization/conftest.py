import pytest
from minipack.pipelines.minimization.pipeline_builder import PipelineBuilder
from minipack.target_estimators import ExponentialDecayEstimator


@pytest.fixture()
def mock_pipeline_builder(interaction_matrices, target_matrices, test_matrices, mock_model):
    builder = PipelineBuilder()

    builder.interaction_matrices = interaction_matrices
    builder.target_matrices = target_matrices
    builder.test_matrices = test_matrices
    builder.model = mock_model

    # Uses default parameters:
    builder.add_minimizer("RandomSelectionMinimizer", params={"metric": {"name": "NDCG", "params": {"K": 100}}})
    builder.add_minimizer("GreedyForwardMinimizer", params={"metric": {"name": "NDCG", "params": {"K": 100}}})
    builder.add_minimizer("GreedyBeamForwardMinimizer", params={"metric": {"name": "NDCG", "params": {"K": 100}}})
    builder.add_target_estimator("ExponentialDecayEstimator", params={"K": 100, "gamma": 0.1})
    builder.add_metric("NDCG", 2)
    builder.add_metric("NDCG", 3)
    builder.save_datasets = False
    builder.save_results = False
    builder.results_directory = "./dummy_dir"
    builder.log_file_path = "./dummy_dir/test.log"

    return builder
