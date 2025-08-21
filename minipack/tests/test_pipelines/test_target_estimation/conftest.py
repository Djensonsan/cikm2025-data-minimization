import pytest
from minipack.metrics import NDCG
from minipack.target_estimators import ExponentialDecayEstimator
from minipack.pipelines.target_estimation.pipeline_builder import PipelineBuilder


@pytest.fixture()
def mock_pipeline_builder(small_random_interaction_matrix, small_random_test_matrix, mock_model):
    builder = PipelineBuilder()

    estimator_1 = ExponentialDecayEstimator(K=128, gamma=0.01)
    estimator_2 = ExponentialDecayEstimator(K=32, gamma=0.8)

    builder.interaction_matrix = small_random_interaction_matrix
    builder.test_matrix = small_random_test_matrix
    builder.model = mock_model
    builder.estimators = [estimator_1, estimator_2]

    # Uses default parameters:
    builder.add_minimizer("RandomSelectionMinimizer", params={"metric": {"name": "NDCG", "params": {"K": 100}}})

    builder.add_metric("NDCG", 2)
    builder.add_metric("NDCG", 3)

    builder.save_datasets = False
    builder.save_results = False
    builder.results_directory = "./dummy_dir"
    builder.log_file_path = "./dummy_dir/test.log"

    return builder
