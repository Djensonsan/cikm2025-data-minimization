import time
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from minipack.minimizers.result import MinimizationResult, MinimizationResultBuilder


def test_add_minimizer_identifier_valid(builder):
    builder.set_minimizer_identifier("test_minimizer")
    assert builder.result.minimizer_identifier == "test_minimizer"


def test_add_minimizer_identifier_invalid(builder):
    with pytest.raises(ValueError, match="minimizer_identifier must be a string"):
        builder.set_minimizer_identifier(123)


def test_add_estimator_identifier_valid(builder):
    builder.set_estimator_identifier("test_estimator")
    assert builder.result.estimator_identifier == "test_estimator"


def test_add_estimator_identifier_invalid(builder):
    with pytest.raises(ValueError, match="estimator_identifier must be a string"):
        builder.set_estimator_identifier(123)


def test_add_batch_id_valid(builder):
    builder.set_batch_id(10)
    assert builder.result.batch_id == 10


def test_add_batch_id_invalid(builder):
    with pytest.raises(ValueError, match="batch_id must be an integer"):
        builder.set_batch_id("invalid_batch_id")


def test_add_input_statistics_valid(builder):
    X = csr_matrix([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    builder.set_input_statistics(X)
    assert builder.result.batch_number_of_users == 3
    assert builder.result.batch_original_number_of_input_interactions == 5
    assert builder.result.average_original_number_of_input_interactions == 5 / 3


def test_add_input_statistics_invalid(builder):
    with pytest.raises(ValueError, match="interaction_matrix must be a scipy.sparse.csr_matrix object"):
        builder.set_input_statistics(np.array([[1, 0], [0, 1]]))


def test_add_target_statistics_valid(builder):
    Y = csr_matrix([[0, 1], [1, 0], [1, 1]])
    builder.set_target_statistics(Y)
    assert builder.result.batch_number_of_target_interactions == 4
    assert builder.result.average_number_of_target_interactions == 4 / 3


def test_add_target_statistics_invalid(builder):
    with pytest.raises(ValueError, match="target_matrix must be a scipy.sparse.csr_matrix object"):
        builder.set_target_statistics(np.array([[1, 0], [0, 1]]))


def test_add_output_statistics_valid(builder):
    X_min = csr_matrix([[1, 0], [0, 0], [1, 1]])
    builder.set_output_statistics(X_min)
    assert builder.result.batch_minimized_number_of_input_interactions == 3


def test_add_output_statistics_invalid(builder):
    with pytest.raises(
        ValueError, match="minimized_matrix must be a scipy.sparse.csr_matrix object"
    ):
        builder.set_output_statistics(np.array([[1, 0], [0, 0]]))


def test_stop_timing_valid_int(builder, X):
    builder.set_input_statistics(X)
    builder.start_timer(user_ids=0)
    time.sleep(0.01)
    builder.stop_timer()
    assert 0.0 < builder.result.per_user_runtimes[0] < 1.0


def test_stop_timing_valid_array(builder, X):
    user_ids = np.array([0, 1, 2])
    builder.set_input_statistics(X)
    builder.start_timer(user_ids)
    time.sleep(0.01)
    builder.stop_timer()
    assert np.all(0 < builder.result.per_user_runtimes[user_ids])
    assert np.all(builder.result.per_user_runtimes[user_ids] < 1.0)


def test_stop_timing_increment_int(builder, X):
    builder.set_input_statistics(X)
    builder.start_timer(user_ids=0, init_value=0.0)
    time.sleep(0.01)
    builder.stop_timer(increment=True)
    assert 0 < builder.result.per_user_runtimes[0] < 1


def test_stop_timing_increment_array(builder, X):
    user_ids = np.array([0, 1, 2])
    builder.set_input_statistics(X)
    builder.start_timer(user_ids, init_value=0.0)
    time.sleep(0.01)
    builder.stop_timer(increment=True)
    assert np.all(0 < builder.result.per_user_runtimes[user_ids])
    assert np.all(builder.result.per_user_runtimes[user_ids] < 1.0)


def test_stop_timing_no_start_time(builder, X):
    builder.set_input_statistics(X)
    time.sleep(0.01)
    with pytest.raises(ValueError, match="Start time is not set."):
        builder.stop_timer()


def test_stop_timing_invalid_user_id(builder, X):
    builder.set_input_statistics(X)
    with pytest.raises(
        ValueError, match="user_ids must be an int or a numpy.ndarray object with dtype int."
    ):
        builder.start_timer("invalid")


def test_stop_timing_invalid_increment(builder, X):
    builder.set_input_statistics(X)
    builder.start_timer(user_ids=0)
    with pytest.raises(ValueError, match="increment must be a boolean."):
        builder.stop_timer(increment="not_a_boolean")


def test_stop_timing_uninitialized_runtimes(builder, X):
    builder.set_input_statistics(X)
    with pytest.raises(ValueError, match="Start time is not set."):
        builder.stop_timer()


def test_add_sample_count_results_valid(builder, X):
    user_ids = np.array([0, 1, 2])
    builder.set_input_statistics(X)
    builder.set_sample_counts(user_ids=user_ids, counts=10)
    assert np.all(builder.result.per_user_sample_counts[user_ids] == 10)


def test_add_sample_count_results_invalid(builder, X):
    builder.set_input_statistics(X)
    with pytest.raises(
        ValueError, match="user_ids must be an int or a numpy.ndarray object with dtype int."
    ):
        builder.set_sample_counts([10, 20, 30], counts=1)  # List instead of numpy array


def test_add_per_user_scores_valid(builder, X):
    builder.set_input_statistics(X)

    user_ids = np.array([0, 1, 2])
    scores = np.array([0.5, 0.6, 0.7])
    builder.set_scores(user_ids, scores)
    assert np.array_equal(builder.result.per_user_scores, scores)


def test_add_per_user_scores_invalid(builder, X):
    builder.set_input_statistics(X)
    user_ids = np.array([0, 1, 2])
    scores = [0.5, 0.6, 0.7]
    with pytest.raises(
        ValueError, match="scores must be a float or a numpy.ndarray object with dtype int."
    ):
        builder.set_scores(user_ids, scores)


def test_add_per_user_performance_threshold_valid(builder):
    thresholds = np.array([0.4, 0.5, 0.6])
    builder.set_per_user_performance_threshold(thresholds)
    assert np.array_equal(builder.result.per_user_performance_threshold, thresholds)


def test_add_per_user_performance_threshold_invalid(builder):
    with pytest.raises(
        ValueError, match="performance_threshold must be a numpy.ndarray object"
    ):
        builder.set_per_user_performance_threshold(
            [0.4, 0.5, 0.6]
        )  # List instead of numpy array


def test_add_timeout_occurred_valid(builder):
    builder.set_timeout(True)
    assert builder.result.timeout_occurred is True


def test_add_timeout_occurred_invalid(builder):
    with pytest.raises(ValueError, match="timeout_occurred must be a boolean"):
        builder.set_timeout("not_a_boolean")


def test_check_readiness_incomplete(builder):
    with pytest.raises(ValueError, match="Minimizer identifier is missing"):
        builder.build()


def test_build_full_result(builder):
    user_ids = np.array([0, 1, 2])
    X = csr_matrix([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    X_min = csr_matrix([[1, 0, 0], [0, 0, 1], [1, 0, 0]])
    Y = csr_matrix([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    sample_counts = np.array([10, 20, 30])
    scores = np.array([0.5, 0.6, 0.7])
    thresholds = np.array([0.4, 0.5, 0.6])

    builder.set_minimizer_identifier("test_minimizer")
    builder.set_estimator_identifier("test_estimator")
    builder.set_batch_id(0)
    builder.set_input_statistics(X)
    builder.set_target_statistics(Y)
    builder.set_per_user_performance_threshold(thresholds)
    builder.set_output_statistics(X_min)
    builder.set_timeout(False)
    builder.start_timer(user_ids=0, init_value=0.0)
    builder.set_scores(user_ids, scores, init_value=0.0)
    builder.set_sample_counts(user_ids, sample_counts, init_value=0.0)

    result = builder.build()

    assert result.minimizer_identifier == "test_minimizer"
    assert result.batch_id == 0
    assert result.batch_number_of_users == 3
    assert result.batch_original_number_of_input_interactions == 5
    assert result.average_original_number_of_input_interactions == 5 / 3
    assert result.batch_number_of_target_interactions == 5
    assert result.average_number_of_target_interactions == 5 / 3
    assert result.batch_number_of_users_processed == 3
    assert result.batch_minimized_number_of_input_interactions == 3
    assert result.average_minimized_number_of_input_interactions == 1
    assert result.timeout_occurred is False
    assert result.batch_percentage_of_users_processed == 1.0
    assert result.batch_constraint_satisfaction == 1.0
    assert result.batch_minimization_ratio == 3 / 5
    assert result.batch_sample_count == 60
    assert result.average_sample_count == 60 / 3
    assert result.average_runtime == 0

def test_update_with_none_sample_counts_and_runtimes():
    builder = MinimizationResultBuilder()
    result = MinimizationResult(
        minimized_matrix=csr_matrix([[1, 0], [0, 1]]),
        timeout_occurred=False,
        per_user_scores=np.array([0.5, 0.6]),
        per_user_sample_counts=np.array([10, 20]),
        per_user_runtimes=np.array([1.0, 2.0])
    )

    builder.update(result)

    assert np.array_equal(builder.result.per_user_scores, np.array([0.5, 0.6]))
    assert np.array_equal(builder.result.per_user_sample_counts, np.array([10, 20]))
    assert np.array_equal(builder.result.per_user_runtimes, np.array([1.0, 2.0]))

def test_update_with_existing_sample_counts_and_runtimes():
    builder = MinimizationResultBuilder()
    builder.result.per_user_sample_counts = np.array([5, 15])
    builder.result.per_user_runtimes = np.array([0.5, 1.5])

    result = MinimizationResult(
        minimized_matrix=csr_matrix([[1, 0], [0, 1]]),
        timeout_occurred=False,
        per_user_scores=np.array([0.5, 0.6]),
        per_user_sample_counts=np.array([10, 20]),
        per_user_runtimes=np.array([1.0, 2.0])
    )

    builder.update(result)

    assert np.array_equal(builder.result.per_user_scores, np.array([0.5, 0.6]))
    assert np.array_equal(builder.result.per_user_sample_counts, np.array([15, 35]))
    assert np.array_equal(builder.result.per_user_runtimes, np.array([1.5, 3.5]))

def test_update_with_timeout():
    builder = MinimizationResultBuilder()
    result = MinimizationResult(
        minimized_matrix=csr_matrix([[1, 0], [0, 1]]),
        timeout_occurred=True,
        per_user_scores=np.array([0.5, 0.6]),
        per_user_sample_counts=np.array([10, 20]),
        per_user_runtimes=np.array([1.0, 2.0])
    )

    builder.update(result)

    assert builder.result.timeout_occurred is True