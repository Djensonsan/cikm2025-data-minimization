import logging
import numpy as np
from scipy.sparse import csr_matrix
from minipack.minimizers.base import BaseMinimizer
from minipack.minimizers.util import sort_lil_matrix
from minipack.minimizers.result import MinimizationResultBuilder

logger = logging.getLogger("minipack")


class EmbeddingSimilaritySelectionMinimizer(BaseMinimizer):
    r"""
    A minimizer that selects the most similar items to a user's history based on model embeddings.

    This class implements a minimization algorithm designed to build a reduced dataset by selecting interactions
    that are most similar to a user's history, as represented by model embeddings. The algorithm iteratively adds
    interactions until a specified performance criterion is satisfied.

    Alternatively, this method could be characterized as picking the items with the highest predicted scores from the user history.

    Formal Definition:
    Let φ represent the user embedding derived from the user's history \( H \), and let \( q_i \) denote the embedding of item \( i \).
    The objective is to select a subset of user interactions \( I \subseteq H \) such that:

    \\[
    I = \underset{I \subseteq H}{\text{argmax}} \; \sum_{i \in I} \phi(H) \cdot q_i
    \\]

    where \( \phi(H) \cdot q_i \) represents the dot product between the user embedding and the item's embedding.
    """
    def _setup(self, interaction_matrix: csr_matrix) -> tuple:
        """
        Prepares the necessary matrices and variables for minimization.
        """
        minimized_matrix = csr_matrix(interaction_matrix.shape, dtype=interaction_matrix.dtype)

        prediction_matrix = self.model._predict(interaction_matrix)

        # Ensure user history remains selectable by adding a small bias.
        bias_matrix = interaction_matrix.multiply(1e-6)

        # Contains the most similar items compared to the user input query
        similarity_matrix = (interaction_matrix.multiply(prediction_matrix) + bias_matrix).tolil()

        # Create a LIL matrix sorted by value (ascending) for efficient retrieval of the most similar items.
        # This differs from the default LIL order (sorted by column index) and may introduce overhead when converting formats.
        # While this optimization is efficient for our use case, it could affect other sparse operations.
        sort_lil_matrix(similarity_matrix, inplace=True)

        # Map new user indices to original user indices
        lookup_original_user_index = np.array(range(interaction_matrix.shape[0]), dtype=int)

        # holds rows that don't meet the performance threshold
        users_below_threshold = np.array(range(interaction_matrix.shape[0]), dtype=int)

        return minimized_matrix, similarity_matrix, lookup_original_user_index, users_below_threshold

    def _minimize(self, interaction_matrix: csr_matrix, target_matrix: csr_matrix) -> MinimizationResultBuilder:
        """Executes the minimization process using an embedding similarity selection strategy.

        Args:
            interaction_matrix (csr_matrix): A sparse matrix representing input data, where rows correspond to users and columns to items.
            target_matrix (csr_matrix): A sparse matrix representing ground truth data, same shape as "interaction_matrix".

        Returns:
            MinimizationResult: An object containing the results of the minimization process.
        """
        minimized_matrix, similarity_matrix, lookup_original_user_idx, users_below_threshold = self._setup(interaction_matrix)

        while True:
            self.result_builder.start_timer(user_ids=users_below_threshold, init_value=0.0)

            # Forward pass: Predict and score
            scores = self._forward(
                input_matrix=minimized_matrix[users_below_threshold],
                history_matrix=interaction_matrix[users_below_threshold],
                target_matrix=target_matrix[users_below_threshold],
            )
            self.result_builder.add_sample_count(
                user_ids=users_below_threshold, counts=1, init_value=0.0
            )
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
                item_idx = similarity_matrix.rows[user_idx_original].pop()
                similarity_matrix.data[user_idx_original].pop()
                minimized_matrix[user_idx_original, item_idx] = 1
                lookup_original_user_idx[user_idx_new] = user_idx_original

            # Convert back to CSR format for efficient math. operations
            minimized_matrix = minimized_matrix.tocsr()

            self.result_builder.stop_timer(increment=True)

            if self.timeout_manager.has_timed_out():
                break

        self.result_builder.set_output_statistics(minimized_matrix)
        return self.result_builder