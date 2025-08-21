import os
import shutil
import pytest
import numpy as np
from scipy.sparse import csr_matrix

from recpack.algorithms.base import TorchMLAlgorithm

from minipack.algorithms.mult_vae import MultVAE
from minipack.algorithms.rec_vae import RecVAE
from minipack.algorithms.nearest_neighbour import ItemKNN, ItemPNN
from minipack.algorithms.ease import EASE
from minipack.algorithms.factorization_item_similarity import (
    SVDItemToItem,
    NMFItemToItem,
)
from minipack.metrics import NDCG
from minipack.pipelines.entries import MinimizerEntry
from minipack.pipelines.minimization.pipeline_builder import PipelineBuilder

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
        builder.interaction_matrices = random_tensor

    assert (
        error.value.args[0]
        == f"Argument should be list of CSR Matrix, not {type(random_tensor)}"
    )


def test_pipeline_builder_bad_groundtruth_data(random_tensor):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    with pytest.raises(TypeError) as error:
        builder.target_matrices = random_tensor

    assert (
        error.value.args[0]
        == f"Argument should be list of CSR Matrix, not {type(random_tensor)}"
    )



def test_pipeline_builder_no_input_data():
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    with pytest.raises(RuntimeError) as error:
        builder.build()

    assert error.value.args[0] == "No input data available, can't construct pipeline"


def test_pipeline_builder_no_groundtruth_data(interaction_matrices):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    builder.interaction_matrices = interaction_matrices
    assert len(builder.interaction_matrices) == len(interaction_matrices)

    # Build pipeline without test data
    with pytest.raises(RuntimeError) as error:
        builder.build()

    assert (
        error.value.args[0]
        == "No ground truth data available, can't construct pipeline. You must either explicitly provide ground truth data or specify a target estimator."
    )


def test_pipeline_builder_no_model(interaction_matrices, target_matrices):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    builder.interaction_matrices = interaction_matrices
    builder.target_matrices = target_matrices
    assert len(builder.interaction_matrices) == len(interaction_matrices)

    # Build pipeline without test data
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


def test_pipeline_builder_no_algorithms(interaction_matrices, target_matrices, mock_model):
    builder = PipelineBuilder()
    builder.interaction_matrices = interaction_matrices
    builder.target_matrices = target_matrices
    builder.model = mock_model
    # Build pipeline without algorithms
    builder.add_metric("NDCG", 20)
    with pytest.raises(RuntimeError) as error:
        builder.build()
    assert error.value.args[0] == "No minimization algorithms specified, can't construct pipeline"

def test_pipeline_builder_full_success(
        interaction_matrices, target_matrices, mock_model
):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    builder.interaction_matrices = interaction_matrices
    builder.target_matrices = target_matrices
    assert len(builder.interaction_matrices) == len(interaction_matrices)

    builder.model = mock_model
    builder.results_directory = "dummy_dir"
    builder.log_file_path = "dummy_log_file"

    pipeline = builder.build()

    assert len(pipeline.minimizer_entries) == 1

def test_pipeline_builder_bad_test_data(random_tensor):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")
    builder.add_metric("NDCG", 20)

    # Build pipeline without test data
    with pytest.raises(TypeError) as error:
        builder.test_matrices = random_tensor

    assert (
        error.value.args[0]
        == f"Argument should be list of CSR Matrix, not {type(random_tensor)}"
    )

def test_pipeline_mismatching_shapes_test(interaction_matrices, target_matrices, mock_model):
    builder = PipelineBuilder()
    builder.add_minimizer("RandomSelectionMinimizer")

    builder.interaction_matrices = interaction_matrices
    # One bin has the wrong shape
    target_dense_matrix = np.random.randint(2, size=(99, 11))
    target_sparse_matrix = csr_matrix(target_dense_matrix)
    target_matrices[0] = target_sparse_matrix

    builder.target_matrices = target_matrices
    builder.results_directory = "dummy_dir"

    builder.model = mock_model

    with pytest.raises(RuntimeError) as error:
        builder.build()
    assert (
        error.value.args[0]
        == "Shape mismatch between csr matrices in input data and ground truth data, can't construct pipeline"
    )

def test_pipeline_duplicate_metric(
        interaction_matrices, target_matrices, test_matrices, mock_model
):
    builder = PipelineBuilder()
    builder.add_metric("NDCG", 20)
    builder.add_metric("NDCG", 20)
    builder.add_metric("NDCG", 20)
    builder.add_minimizer("RandomSelectionMinimizer")

    builder.interaction_matrices = interaction_matrices
    builder.target_matrices = target_matrices
    builder.test_matrices = test_matrices
    builder.model = mock_model
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
def test_model_compatibility(large_random_interaction_matrix, large_random_target_matrix, large_random_test_matrix,
                             interaction_matrices,
                             target_matrices, model):
    # Fitting on random data, using default values
    model = model()

    if isinstance(model, TorchMLAlgorithm):
        model.fit(large_random_interaction_matrix, (large_random_target_matrix, large_random_test_matrix))
        model.best_model = None
    else:
        model.fit(large_random_interaction_matrix)

    builder = PipelineBuilder()

    builder.interaction_matrices = interaction_matrices
    builder.target_matrices = target_matrices
    builder.model = model

    builder.add_minimizer("RandomSelectionMinimizer", params={"metric": {"name": "NDCG", "params": {"K": 100}}, "timeout": 0.5})
    builder.add_minimizer("GreedyForwardMinimizer", params={"metric": {"name": "NDCG", "params": {"K": 100}}, "timeout": 0.5})
    builder.add_minimizer("GreedyBeamForwardMinimizer", params={"metric": {"name": "NDCG", "params": {"K": 100}}, "timeout": 0.5})

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

        num_bins = len(pipeline.interaction_matrices)
        num_minimizers = len(pipeline.minimizer_entries)
        assert len(pipeline.batch_results) == num_bins * num_minimizers

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