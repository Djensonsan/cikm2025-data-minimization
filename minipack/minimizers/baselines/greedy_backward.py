import logging
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from minipack.minimizers.base import BaseMinimizer
from minipack.minimizers.util import repeat_lil_vector
from minipack.minimizers.result import MinimizationResultBuilder

logger = logging.getLogger("minipack")


class GreedyBackwardMinimizer(BaseMinimizer):
    """A minimizer that employs a greedy forward selection strategy.

    This class implements a minimization algorithm that iteratively removes interactions
    to include in the minimized dataset using a greedy approach. The algorithm aims to
    optimize a metric in each step of the selection process.

    The class inherits from BaseMinimizer and implements the `_minimize` method.
    """

    def _minimize(
        self, interaction_matrix: csr_matrix, target_matrix: csr_matrix
    ) -> MinimizationResultBuilder:
        """Executes the minimization process using a greedy forward selection strategy.

        Args:
            interaction_matrix (csr_matrix): A sparse matrix representing input data, where rows correspond to users and columns to items.
            target_matrix (csr_matrix): A sparse matrix representing ground truth data, structured similarly to X.

        Returns:
            MinimizationResult: Results of the minimization process.
        """
        interaction_matrix = interaction_matrix.tolil()
        target_matrix = target_matrix.tolil()
        minimized_matrix = lil_matrix(interaction_matrix.shape, dtype=interaction_matrix.dtype)
        for user_idx, (interaction_vector, target_vector) in enumerate(zip(interaction_matrix, target_matrix)):
            self.result_builder.start_timer(user_ids=user_idx)
            self.result_builder.set_sample_counts(user_ids=user_idx, counts=0)

            map_candidate_to_item = np.array(range(interaction_vector.getnnz()), dtype=int)

            # Faster to pre-calculate and slice later
            repeated_interaction_matrix = repeat_lil_vector(vector=interaction_vector,times=interaction_vector.getnnz()).tocsr()
            repeated_target_matrix = repeat_lil_vector(vector=target_vector, times=interaction_vector.getnnz()).tocsr()

            # Compute score for full user history first before starting minimization loop:
            interaction_vector = interaction_vector.tocsr()
            target_vector = target_vector.tocsr()

            scores = self._forward(
                input_matrix=interaction_vector,
                history_matrix=interaction_vector,
                target_matrix=target_vector
            )
            self.result_builder.add_sample_count(
                user_ids=user_idx, counts=interaction_vector.shape[0]
            )
            self.result_builder.set_scores(
                user_ids=user_idx, scores=scores[0]
            )

            interaction_vector = interaction_vector.tolil()

            while interaction_vector.getnnz() > 0:
                candidates_matrix = repeat_lil_vector(vector=interaction_vector, times=interaction_vector.getnnz())

                # Update column index map and interaction candidates
                for i, item_idx in enumerate(interaction_vector.nonzero()[1]):
                    candidates_matrix[i, item_idx] = 0
                    map_candidate_to_item[i] = item_idx

                sliced_target_matrix = repeated_target_matrix[: candidates_matrix.shape[0], :]
                sliced_interaction_matrix = repeated_interaction_matrix[: candidates_matrix.shape[0], :]
                candidates_matrix = candidates_matrix.tocsr()

                scores = self._forward(
                    input_matrix=candidates_matrix,
                    history_matrix=sliced_interaction_matrix,
                    target_matrix=sliced_target_matrix
                )
                self.result_builder.add_sample_count(
                    user_ids=user_idx, counts=candidates_matrix.shape[0]
                )

                # Find the best interaction to remove
                best_candidate_index = np.argmax(scores)
                best_item_index = map_candidate_to_item[best_candidate_index]

                if scores[best_candidate_index] >= self.thresholds[user_idx]:
                    self.result_builder.set_scores(
                        user_ids=user_idx, scores=scores[best_candidate_index]
                    )
                    interaction_vector[0, best_item_index] = 0
                else:
                    break

            # minimized interactions for this user
            minimized_matrix[user_idx] = interaction_vector

            self.result_builder.stop_timer()
            if self.timeout_manager.has_timed_out():
                # If timeout, return the processed part of the matrix
                minimized_matrix = minimized_matrix[: (user_idx + 1), :]
                break

        minimized_matrix = minimized_matrix.tocsr()
        self.result_builder.set_output_statistics(minimized_matrix)
        return self.result_builder