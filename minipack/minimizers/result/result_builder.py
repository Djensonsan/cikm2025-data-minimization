import time
import numpy as np
from scipy.sparse import csr_matrix
from minipack.minimizers.result.result import MinimizationResult
from minipack.minimizers.util import pad_with_nan

class MinimizationResultBuilder:
    def __init__(self):
        self.result = MinimizationResult()
        self.result.timeout_occurred = False

    def set_minimizer_identifier(self, minimizer_identifier: str):
        """
        Add the identifier of the minimizer that was used to generate the results.

        Mandatory Attribute.
        """
        if not isinstance(minimizer_identifier, str):
            raise ValueError("minimizer_identifier must be a string.")
        self.result.minimizer_identifier = minimizer_identifier

    def set_estimator_identifier(self, estimator_identifier: str):
        """
        Add the identifier of the estimator that was used to generate the results.

        Optional Attribute.
        """
        if not isinstance(estimator_identifier, str):
            raise ValueError("estimator_identifier must be a string.")
        self.result.estimator_identifier = estimator_identifier

    def set_timeout_occurred(self, timeout_occurred: bool):
        """
        Add whether a timeout occurred during the minimization process.

        Mandatory Attribute.
        """
        if not isinstance(timeout_occurred, bool):
            raise ValueError("timeout_occurred must be a boolean.")
        self.result.timeout_occurred = timeout_occurred

    def set_batch_id(self, batch_id: int):
        """
        Add the batch_id of the batch that was minimized.

        Optional Attribute.
        """
        if not isinstance(batch_id, int):
            raise ValueError("batch_id must be an integer.")
        self.result.batch_id = batch_id

    def set_input_statistics(self, interaction_matrix: csr_matrix):
        """
        Add the input statistics of the minimization process.

        Mandatory Attribute.
        """
        if not isinstance(interaction_matrix, csr_matrix):
            raise ValueError("interaction_matrix must be a scipy.sparse.csr_matrix object.")

        self.result.batch_number_of_users = interaction_matrix.shape[0]
        self.result.batch_original_number_of_input_interactions = interaction_matrix.nnz
        self.result.per_user_original_input_interactions = interaction_matrix.getnnz(axis=1)
        self.result.average_original_number_of_input_interactions = interaction_matrix.nnz / interaction_matrix.shape[0]

    def set_target_statistics(self, target_matrix: csr_matrix):
        """
        Add the target statistics of the minimization process.

        Optional Attribute.
        """
        if not isinstance(target_matrix, csr_matrix):
            raise ValueError("target_matrix must be a scipy.sparse.csr_matrix object.")
        self.result.batch_number_of_target_interactions = target_matrix.nnz
        self.result.per_user_target_interactions = target_matrix.getnnz(axis=1)
        self.result.average_number_of_target_interactions = target_matrix.nnz / target_matrix.shape[0]

    def set_per_user_performance_threshold(
        self, per_user_performance_threshold: np.ndarray
    ):
        """
        Add the per-user performance threshold of the minimization process.

        Mandatory Attribute.
        """
        if not isinstance(per_user_performance_threshold, np.ndarray):
            raise ValueError("performance_threshold must be a numpy.ndarray object.")
        self.result.per_user_performance_threshold = per_user_performance_threshold

    def set_output_statistics(self, minimized_matrix: csr_matrix):
        """
        Add the output statistics of the minimization process.

        Mandatory Attribute.
        """
        if not isinstance(minimized_matrix, csr_matrix):
            raise ValueError("minimized_matrix must be a scipy.sparse.csr_matrix object.")
        minimized_matrix.eliminate_zeros()
        self.result.minimized_matrix = minimized_matrix
        self.result.batch_minimized_number_of_input_interactions = minimized_matrix.nnz
        # Convention: If a minimizer receives a timeout, it should return a partial matrix.
        self.result.batch_number_of_users_processed = minimized_matrix.shape[0]
        self.result.average_minimized_number_of_input_interactions = (
                minimized_matrix.nnz / minimized_matrix.shape[0]
        )

    def set_timeout(self, timeout_occurred: bool):
        """
        Add whether a timeout occurred during the minimization process.

        Mandatory Attribute.
        """
        if not isinstance(timeout_occurred, bool):
            raise ValueError("timeout_occurred must be a boolean.")
        self.result.timeout_occurred = timeout_occurred

    def set_scores(self, user_ids: [int, np.integer, np.ndarray], scores: [float, np.floating, np.ndarray], init_value=np.NaN):
        """
        Set the score for a given user or users. This function assigns a specified
        score to a particular user ID or a set of user IDs.
        """
        if self.result.per_user_scores is None:
            if not hasattr(self.result, "batch_number_of_users") or self.result.batch_number_of_users is None:
                raise ValueError("Input statistics are missing. Required for initializing result attributes.")
            number_of_users = self.result.batch_number_of_users
            self.result.per_user_scores = np.full(number_of_users, init_value, dtype=float)
        if not isinstance(user_ids, (int, np.integer, np.ndarray)):
            raise ValueError("user_ids must be an int or a numpy.ndarray object with dtype int.")
        if not isinstance(scores, (int, np.floating, np.ndarray)):
            raise ValueError("scores must be a float or a numpy.ndarray object with dtype int.")
        if isinstance(user_ids, np.ndarray) and isinstance(scores, np.ndarray):
            if user_ids.shape != scores.shape:
                raise ValueError(
                    f"Shape mismatch: user_ids has shape {user_ids.shape}, but value has shape {scores.shape}.")
        self.result.per_user_scores[user_ids] = scores

    def set_sample_counts(self, user_ids: [int, np.integer, np.ndarray], counts: [int, np.integer, np.ndarray], init_value=np.NaN):
        """
        Updates the per-user sample counts for a given user ID. It either sets the count to a
        specified value or increments the existing count by the given value.

        If the input `user_id` is an integer, it updates the single user's count. If the input
        is a numpy.ndarray, it updates the sample counts for all specified user IDs.
        """
        if self.result.per_user_sample_counts is None:
            if not hasattr(self.result, "batch_number_of_users") or self.result.batch_number_of_users is None:
                raise ValueError("Input statistics are missing. Required for initializing result attributes.")
            number_of_users = self.result.batch_number_of_users
            self.result.per_user_sample_counts = np.full(number_of_users, init_value, dtype=float)
        if not isinstance(user_ids, (int, np.integer, np.ndarray)):
            raise ValueError("user_ids must be an int or a numpy.ndarray object with dtype int.")
        if not isinstance(counts, (int, np.integer, np.ndarray)):
            raise ValueError("count must be an int or a numpy.ndarray object with dtype int.")
        self.result.per_user_sample_counts[user_ids] = counts

    def add_sample_count(self, user_ids: [int, np.integer, np.ndarray], counts: [int, np.integer, np.ndarray], init_value=np.NaN):
        if self.result.per_user_sample_counts is None:
            if not hasattr(self.result, "batch_number_of_users") or self.result.batch_number_of_users is None:
                raise ValueError("Input statistics are missing. Required for initializing result attributes.")
            number_of_users = self.result.batch_number_of_users
            self.result.per_user_sample_counts = np.full(number_of_users, init_value, dtype=float)
        if not isinstance(user_ids, (int, np.integer, np.ndarray)):
            raise ValueError("user_ids must be an int or a numpy.ndarray object with dtype int.")
        if not isinstance(counts, (int, np.integer, np.ndarray)):
            raise ValueError("count must be an int or a numpy.ndarray object with dtype int.")
        self.result.per_user_sample_counts[user_ids] += counts

    def start_timer(self, user_ids: [int, np.integer, np.ndarray] = None, init_value=np.NaN, ):
        """
        Starts the timing for a specific user or set of users. This method initializes
        the tracking by setting the user ID being tracked and records the start time.
        The user ID must be specified, otherwise, an exception is raised.
        """
        if self.result.per_user_runtimes is None:
            if not hasattr(self.result, "batch_number_of_users") or self.result.batch_number_of_users is None:
                raise ValueError("Input statistics are missing. Required for initializing result attributes.")
            number_of_users = self.result.batch_number_of_users
            self.result.per_user_runtimes = np.full(number_of_users, init_value, dtype=float)
        if user_ids is None:
            raise ValueError("user_id must be specified.")
        if not isinstance(user_ids, (int, np.integer, np.ndarray)):
            raise ValueError("user_ids must be an int or a numpy.ndarray object with dtype int.")
        self.tracked_user_ids = user_ids
        self.start_time = time.perf_counter()

    def stop_timer(self, increment=False):
        """
        Stops the timing operation for a given user ID or an array of user IDs.
        Calculates the runtime by subtracting the start time from the current
        time using a performance counter. Depending on the `increment` flag,
        either increments the runtime in `per_user_runtimes` or directly sets it.
        """
        if not isinstance(increment, bool):
            raise ValueError("increment must be a boolean.")
        if not hasattr(self, "start_time") or (self.start_time is None):
            raise ValueError("Start time is not set.")
        runtime = time.perf_counter() - self.start_time
        if increment:
            if isinstance(self.tracked_user_ids, (int, np.integer)):
                self.result.per_user_runtimes[self.tracked_user_ids] += runtime
            else:
                self.result.per_user_runtimes[self.tracked_user_ids] += runtime / len(self.tracked_user_ids)
        else:
            self.result.per_user_runtimes[self.tracked_user_ids] = runtime

    def _check_readiness(self):
        """Check if the builder is ready to create a full results object."""
        if self.result.minimizer_identifier is None:
            raise ValueError("Minimizer identifier is missing.")
        if self.result.batch_original_number_of_input_interactions is None:
            raise ValueError("Input statistics are missing.")
        if self.result.batch_minimized_number_of_input_interactions is None:
            raise ValueError("Output statistics are missing.")
        if self.result.per_user_runtimes is None:
            raise ValueError("Runtime results are missing.")
        if self.result.per_user_sample_counts is None:
            raise ValueError("Sample count results are missing.")
        if self.result.per_user_scores is None:
            raise ValueError("Per user scores are missing.")
        if self.result.per_user_performance_threshold is None:
            raise ValueError("Per user performance thresholds are missing.")
        if self.result.timeout_occurred is None:
            raise ValueError("Timeout occurrence is missing.")

    def _compute_constraint_satisfaction(self):
        # Timeout logic: Constraint satisfaction set to NaN if user had no scores.
        nan_mask = np.isnan(self.result.per_user_scores)
        comparison = (
            self.result.per_user_scores >= self.result.per_user_performance_threshold
        )
        self.result.per_user_constraint_satisfaction = np.where(
            nan_mask, np.nan, comparison
        )
        self.result.batch_constraint_satisfaction = np.nanmean(
            self.result.per_user_constraint_satisfaction
        )

    def _compute_batch_percentage_of_users_processed(self):
        if self.result.batch_number_of_users != 0:
            self.result.batch_percentage_of_users_processed = (
                    self.result.batch_number_of_users_processed
                    / self.result.batch_number_of_users
            )
        else:
            self.result.batch_percentage_of_users_processed = np.NaN  # Assign NaN if division by zero

    def _compute_batch_minimization_ratio(self):
        if self.result.batch_original_number_of_input_interactions != 0:
            self.result.batch_minimization_ratio = (
                    self.result.batch_minimized_number_of_input_interactions
                    / self.result.batch_original_number_of_input_interactions
            )
        else:
            self.result.batch_minimization_ratio = np.NaN  # Handle division by zero

    def _compute_per_user_minimized_input_interactions(self):
        per_user_minimized_input_interactions = self.result.minimized_matrix.getnnz(axis=1)
        self.result.per_user_minimized_input_interactions = pad_with_nan(
            per_user_minimized_input_interactions,
            self.result.batch_number_of_users
        )

    def _compute_per_user_minimization_ratios(self):
        # Handle division by zero:
        self.result.per_user_minimization_ratios = np.divide(
            self.result.per_user_minimized_input_interactions,
            self.result.per_user_original_input_interactions,
            out = np.full_like(self.result.per_user_minimized_input_interactions, np.nan),
            where = self.result.per_user_original_input_interactions != 0
        )

    def _compute_derivative_sample_counts(self):
        # Handle division by zero:
        self.result.batch_sample_count = np.nansum(self.result.per_user_sample_counts)
        if self.result.batch_number_of_users_processed != 0:
            self.result.average_sample_count = (
                self.result.batch_sample_count / self.result.batch_number_of_users_processed
            )
        else:
            self.result.average_sample_count = np.NaN # Handle division by zero

    def _compute_derivative_runtimes(self):
        self.result.batch_runtime = np.nansum(self.result.per_user_runtimes)
        if self.result.batch_number_of_users_processed != 0:
            self.result.average_runtime = (
                self.result.batch_runtime / self.result.batch_number_of_users_processed
            )
        else:
            self.result.average_runtime = np.NaN # Handle division by zero

    def build(self) -> MinimizationResult:
        self._check_readiness()

        self._compute_batch_percentage_of_users_processed()
        self._compute_per_user_minimized_input_interactions()
        self._compute_batch_minimization_ratio()
        self._compute_per_user_minimization_ratios()
        self._compute_constraint_satisfaction()
        self._compute_derivative_sample_counts()
        self._compute_derivative_runtimes()

        if self.result.timeout_occurred:
            self._compute_per_user_minimization_ratios()

        return self.result

    def update(self, result: MinimizationResult):
        self.set_output_statistics(result.minimized_matrix)
        self.set_timeout(result.timeout_occurred)
        self.result.per_user_scores = result.per_user_scores

        if self.result.per_user_sample_counts is None:
            self.result.per_user_sample_counts = result.per_user_sample_counts
        else:
            self.result.per_user_sample_counts = self.result.per_user_sample_counts + result.per_user_sample_counts

        if self.result.per_user_runtimes is None:
            self.result.per_user_runtimes = result.per_user_runtimes
        else:
            self.result.per_user_runtimes = self.result.per_user_runtimes + result.per_user_runtimes
