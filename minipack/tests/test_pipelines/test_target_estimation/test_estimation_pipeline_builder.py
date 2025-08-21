import os
import pytest
import shutil
from minipack.algorithms.mult_vae import MultVAE
from minipack.algorithms.rec_vae import RecVAE
from recpack.algorithms.base import TorchMLAlgorithm

from minipack.algorithms.nearest_neighbour import ItemKNN, ItemPNN
from minipack.algorithms.ease import EASE
from minipack.algorithms.factorization_item_similarity import (
    SVDItemToItem,
    NMFItemToItem,
)
from minipack.metrics import NDCG
from minipack.target_estimators import ExponentialDecayEstimator
from minipack.pipelines.entries import MinimizerEntry
from minipack.pipelines.target_estimation.pipeline_builder import PipelineBuilder

def test_pipeline_builder_empty():
    builder = PipelineBuilder()

    # Build empty pipeline
    with pytest.raises(RuntimeError) as error:
        builder.build()
    assert error.value.args[0] == "No input data available, can't construct pipeline"


def test_pipeline_builder_bad_input_data(random_tensor):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    with pytest.raises(TypeError) as error:
        builder.interaction_matrix = random_tensor

    assert (
        error.value.args[0]
        == f"Argument should be a CSR Matrix, not {type(random_tensor)}"
    )

def test_pipeline_builder_bad_estimators():
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    with pytest.raises(ValueError) as error:
        builder.add_target_estimator("RandomEstimator")


def test_pipeline_builder_no_input_data():
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    with pytest.raises(RuntimeError) as error:
        builder.build()

    assert error.value.args[0] == "No input data available, can't construct pipeline"


def test_pipeline_builder_no_model(small_random_interaction_matrix):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    builder.interaction_matrix = small_random_interaction_matrix

    with pytest.raises(RuntimeError) as error:
        builder.build()

    assert (
        error.value.args[0]
        == "No recommendation model specified, can't construct pipeline"
    )

def test_pipeline_builder_metrics():
    builder = PipelineBuilder()
    # Add 1 or multiple metrics
    builder.add_metric("NDCG", 20)
    assert len(builder.metric_entries) == 1
    builder.add_metric("NDCG", [10, 30])
    assert len(builder.metric_entries) == 3


def test_pipeline_builder_no_minimizers(small_random_interaction_matrix, mock_model):
    builder = PipelineBuilder()
    builder.interaction_matrix = small_random_interaction_matrix
    builder.model = mock_model
    # Build pipeline without algorithms
    builder.add_metric("NDCG", 20)
    with pytest.raises(RuntimeError) as error:
        builder.build()
    assert error.value.args[0] == "No minimization algorithms specified, can't construct pipeline"

def test_pipeline_builder_no_estimators(small_random_interaction_matrix, mock_model):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    builder.interaction_matrix = small_random_interaction_matrix
    builder.model = mock_model
    builder.add_minimizer("RandomSelectionMinimizer")

    with pytest.raises(RuntimeError) as error:
        builder.build()

    assert error.value.args[0] == "No ground truth estimators specified, can't construct pipeline"

def test_pipeline_builder_full_success(
        small_random_interaction_matrix, mock_model
):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    builder.interaction_matrix = small_random_interaction_matrix

    builder.model = mock_model
    builder.estimators = [ExponentialDecayEstimator(K=128, gamma=0.8)]
    builder.results_directory = "dummy_dir"
    builder.log_file_path = "dummy_log_file"

    pipeline = builder.build()

    assert len(pipeline.minimization_entries) == 1

def test_pipeline_builder_bad_test_data(random_tensor):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")
    builder.add_metric("NDCG", 20)

    # Build pipeline without test data
    with pytest.raises(TypeError) as error:
        builder.test_matrix = random_tensor

    assert (
        error.value.args[0]
        == f"Argument should be a CSR Matrix, not {type(random_tensor)}"
    )

def test_pipeline_duplicate_metric(
        small_random_interaction_matrix, small_random_target_matrix, small_random_test_matrix, mock_model
):
    builder = PipelineBuilder()
    builder.add_metric("NDCG", 20)
    builder.add_metric("NDCG", 20)
    builder.add_metric("NDCG", 20)
    builder.add_minimizer("RandomSelectionMinimizer")

    builder.interaction_matrix = small_random_interaction_matrix
    builder.test_matrix = small_random_test_matrix
    builder.model = mock_model
    builder.estimators = [ExponentialDecayEstimator(K=128, gamma=0.8)]
    builder.results_directory = "dummy_dir"
    builder.log_file_path = "dummy_log_file"

    assert len(builder.metric_entries) == 1
    pipeline = builder.build()

    assert len(pipeline.metric_entries) == 1

# A basic integration test to assure the components work together
@pytest.mark.parametrize(
    "model",
    [
        MultVAE,
        RecVAE,
        ItemKNN,
        ItemPNN,
        EASE,
        NMFItemToItem,
        SVDItemToItem,
    ],
)
def test_model_compatibility(large_random_interaction_matrix, large_random_target_matrix, large_random_test_matrix, model):
    # Fitting on random data, using default values
    model = model()

    if isinstance(model, TorchMLAlgorithm):
        model.fit(large_random_interaction_matrix, (large_random_target_matrix, large_random_test_matrix))
        model.best_model = None
    else:
        model.fit(large_random_interaction_matrix)

    builder = PipelineBuilder()

    builder.interaction_matrix = large_random_interaction_matrix
    builder.model = model

    builder.add_target_estimator("ExponentialDecayEstimator", grid={"K": [1, 2], "gamma": [0.8]})
    builder.add_minimizer("RandomSelectionMinimizer", params={"metric": {"name": "NDCG", "params": {"K": 100}}, "timeout": 0.5})

    builder.add_metric("NDCG", 2)
    builder.add_metric("NDCG", 3)
    builder.save_datasets = True
    builder.save_results = True
    builder.results_directory = "./dummy_dir"
    builder.log_file_path = "./dummy_dir/dummy_log_file.log"

    # Build the pipeline & run
    pipeline = builder.build()

    try:
        pipeline.run(local_mode=True)

        assert pipeline.batch_results is not None
        assert pipeline.user_results is not None

        results_columns = pipeline.batch_results.columns
        assert "minimizer_identifier" in results_columns
        assert "estimator_identifier" in results_columns
        assert "batch_id" in results_columns

        results_columns = pipeline.user_results.columns
        assert "minimizer_identifier" in results_columns
        assert "estimator_identifier" in results_columns
        assert "batch_id" in results_columns

        assert os.path.isdir(pipeline.results_directory)
        assert os.path.isfile(pipeline.batch_results_file_path)
        assert os.path.isfile(pipeline.user_results_file_path)
    finally:
        shutil.rmtree(pipeline.results_directory)
