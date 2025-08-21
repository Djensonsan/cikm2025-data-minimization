import logging
import numpy as np
from scipy.sparse import csr_matrix
from minipack.minimizers.base import BaseMinimizer
from minipack.minimizers.util import sort_lil_matrix
from minipack.minimizers.result import MinimizationResultBuilder

logger = logging.getLogger("minipack")


class MostPopularSelectionMinimizer(BaseMinimizer):
    """A minimizer that employs a most popular selection strategy.

    This class implements a minimization algorithm that incrementally selects the most popular interactions
    to include in the minimized dataset. The algorithm iteratively adds interactions
    until a specified performance criterion is met.

    The class inherits from BaseMinimizer and implements the `minimize` method.
    """
    def _setup(self, X: csr_matrix) -> tuple:
        """
        Prepares the necessary matrices and variables for minimization.
        """
        minimized_matrix = csr_matrix(X.shape, dtype=X.dtype)

        # a matrix with the popularity of each item
        item_pop = np.array(X.sum(axis=0)).flatten()

        # contains the popularity for each user history
        popularity_matrix = X.multiply(item_pop).tolil()

        # Create a LIL matrix sorted by value (ascending) for efficient retrieval of the most similar item.
        # This differs from the default LIL order (sorted by column index) and may introduce overhead or side effects in other sparse operations.
        # Since it's used for a single purpose, this approach remains efficient.
        sort_lil_matrix(popularity_matrix, inplace=True)

        # a map from new row indices (index) to original indices (value)
        lookup_original_user_index = np.array(range(X.shape[0]), dtype=int)

        # holds rows that don't meet the performance threshold
        users_below_threshold = np.array(range(X.shape[0]), dtype=int)
        return minimized_matrix, popularity_matrix, lookup_original_user_index, users_below_threshold

    def _minimize(self, interaction_matrix: csr_matrix, target_matrix: csr_matrix) -> MinimizationResultBuilder:
        """Executes the minimization process using a most popular selection strategy.

        Args:
            interaction_matrix (csr_matrix): A sparse matrix representing input data, where rows correspond to users and columns to items.
            target_matrix (csr_matrix): A sparse matrix representing ground truth data, structured similarly to X.

        Returns:
            MinimizationResult: An object containing the results of the minimization process.
        """
        minimized_matrix, popularity_matrix, lookup_original_user_idx, users_below_threshold = self._setup(interaction_matrix)

        while True:
            self.result_builder.start_timer(user_ids=users_below_threshold, init_value=0.0)

            # Forward pass: Predict and score
            scores = self._forward(
                input_matrix=minimized_matrix[users_below_threshold],
                history_matrix=interaction_matrix[users_below_threshold],
                target_matrix=target_matrix[users_below_threshold],
            )
            self.result_builder.add_sample_count(user_ids=users_below_threshold, counts=1, init_value=0.0)
            self.result_builder.set_scores(user_ids=users_below_threshold, scores=scores)

            # Identify users still below the threshold
            users_below_threshold = np.where(scores < self.thresholds[users_below_threshold])[0]
            users_below_threshold = lookup_original_user_idx[users_below_threshold]

            if users_below_threshold.size == 0:
                self.result_builder.stop_timer(increment=True)
                break

            # Convert to LIL format for efficient insertions
            minimized_matrix = minimized_matrix.tolil()

            for user_idx_new, user_idx_original in enumerate(users_below_threshold):
                item_idx = popularity_matrix.rows[user_idx_original].pop()
                popularity_matrix.data[user_idx_original].pop()
                minimized_matrix[user_idx_original, item_idx] = 1
                lookup_original_user_idx[user_idx_new] = user_idx_original

            # Convert back to CSR format for further operations
            minimized_matrix = minimized_matrix.tocsr()

            self.result_builder.stop_timer(increment=True)

            if self.timeout_manager.has_timed_out():
                break

        self.result_builder.set_output_statistics(minimized_matrix)
        return self.result_builder