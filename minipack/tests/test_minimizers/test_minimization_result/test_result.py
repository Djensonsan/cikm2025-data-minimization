import numpy as np
import pytest
import pandas as pd
from minipack.minimizers.result.result import MinimizationResult

@pytest.fixture
def result():
    return MinimizationResult(
        minimizer_identifier="minimizer_test",
        estimator_identifier="estimator_test",
        batch_id=101,
        batch_number_of_users=3,
        batch_number_of_users_processed=2,
        batch_percentage_of_users_processed=0.8,
        batch_original_number_of_input_interactions=50,
        batch_minimized_number_of_input_interactions=30,
        average_original_number_of_input_interactions=16.67,
        average_minimized_number_of_input_interactions=10.0,
        batch_minimization_ratio=0.6,
        batch_constraint_satisfaction=0.75,
        batch_runtime=120.0,
        batch_sample_count=500,
        average_runtime=40.0,
        average_sample_count=167,
        timeout_occurred=False,
        per_user_scores=np.array([0.8, 0.7, 0.9]),
        per_user_original_input_interactions=np.array([10, 20, 20]),
        per_user_minimized_input_interactions=np.array([8, 15, 18]),
        per_user_runtimes=np.array([50, 40, 30]),
        per_user_sample_counts=np.array([200, 150, 150]),
        per_user_performance_threshold=np.array([0.6, 0.5, 0.7]),
        per_user_constraint_satisfaction=np.array([1, 0, 1]),
    )

def test_minimization_result_initialization(result):
    assert result.minimizer_identifier == "minimizer_test"
    assert result.batch_id == 101
    assert result.batch_number_of_users == 3
    assert result.timeout_occurred is False
    assert result.batch_runtime == 120.0

def test_batch_results_dataframe_structure(result):
    df = result.batch_results
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 19)  # Should have 1 row and 19 attributes from batch_results
    assert list(df.columns) == [
        "minimizer_identifier",
        "estimator_identifier",
        "batch_id",
        "batch_number_of_users",
        "batch_number_of_users_processed",
        "batch_percentage_of_users_processed",
        "batch_original_number_of_input_interactions",
        "batch_minimized_number_of_input_interactions",
        "batch_number_of_target_interactions",
        "batch_minimization_ratio",
        "batch_runtime",
        "batch_constraint_satisfaction",
        "batch_sample_count",
        "average_original_number_of_input_interactions",
        "average_minimized_number_of_input_interactions",
        "average_number_of_target_interactions",
        "average_runtime",
        "average_sample_count",
        "timeout_occurred",
    ]
    assert df["minimizer_identifier"][0] == "minimizer_test"

def test_user_results_dataframe_structure(result):
    df = result.user_results
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 12)  # 3 users, 12 columns (constant + user-specific attributes)
    assert list(df.columns) == [
        "minimizer_identifier",
        "estimator_identifier",
        "batch_id",
        "per_user_scores",
        "per_user_original_input_interactions",
        "per_user_minimized_input_interactions",
        "per_user_target_interactions",
        "per_user_minimization_ratios",
        "per_user_runtimes",
        "per_user_sample_counts",
        "per_user_performance_threshold",
        "per_user_constraint_satisfaction",
    ]
    assert df["minimizer_identifier"][0] == "minimizer_test"
    assert df["per_user_scores"][0] == 0.8
    assert df["per_user_original_input_interactions"][1] == 20
    assert df["per_user_runtimes"][2] == 30

def test_batch_results_values(result):
    df = result.batch_results
    assert df["batch_runtime"][0] == 120.0
    assert df["batch_sample_count"][0] == 500
    assert df["average_runtime"][0] == 40.0
    assert df["average_sample_count"][0] == 167
    assert df["batch_minimization_ratio"][0] == 0.6

def test_user_results_values(result):
    df = result.user_results
    assert df["per_user_scores"].iloc[0] == 0.8
    assert df["per_user_minimized_input_interactions"].iloc[2] == 18
    assert df["per_user_sample_counts"].iloc[1] == 150
    assert df["per_user_constraint_satisfaction"].iloc[2] == 1

def test_batch_results_missing_attributes():
    result = MinimizationResult(minimizer_identifier="test_minimizer")
    df = result.batch_results
    assert isinstance(df, pd.DataFrame)
    assert df["minimizer_identifier"][0] == "test_minimizer"
    assert pd.isna(df["batch_runtime"][0])  # Missing attributes should be NaN

def test_user_results_missing_attributes():
    result = MinimizationResult(minimizer_identifier="test_minimizer", per_user_scores=[0.9, 0.8])
    df = result.user_results
    assert df.shape == (2, 12)  # Only 2 rows, as we only have per_user_scores
    assert df["minimizer_identifier"].iloc[0] == "test_minimizer"
    assert pd.isna(df["per_user_sample_counts"].iloc[0])  # Missing attributes should be NaN