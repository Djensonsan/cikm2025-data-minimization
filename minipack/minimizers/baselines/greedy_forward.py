import logging
import numpy as np
from recpack.algorithms import Algorithm
from recpack.metrics.base import Metric
from scipy.sparse import csr_matrix, lil_matrix
from minipack.minimizers.base import BaseMinimizer
from minipack.minimizers.util import repeat_lil_vector
from minipack.minimizers.result import MinimizationResultBuilder

logger = logging.getLogger("minipack")


class GreedyForwardMinimizer(BaseMinimizer):
    """A minimizer that employs a greedy forward selection strategy.

    This class implements a minimization algorithm that iteratively selects interactions
    to include in the minimized dataset using a greedy approach. The algorithm aims to
    optimize a metric in each step of the selection process.

    The class inherits from BaseMinimizer and implements the `_minimize` method.
    """
    def __init__(
        self,
        metric: Metric = None,
        model: Algorithm = None,
        eta: float = 0.9,
        max_size: int = None,
        remove_history: bool = True,
        timeout=float("inf"),
        timeout_manager=None,
    ):
        super().__init__(metric, model, eta, remove_history, timeout, timeout_manager)
        self.max_size = max_size

    def _minimize(self, interaction_matrix: csr_matrix, target_matrix: csr_matrix) -> MinimizationResultBuilder:
        """Executes the minimization process using a greedy forward selection strategy.

        Args:
            interaction_matrix (csr_matrix): A sparse matrix representing input data, where rows correspond to users and columns to items.
            target_matrix (csr_matrix): A sparse matrix representing ground truth data, structured similarly to X.

        Returns:
            MinimizationResult: An object containing the results of the minimization process.
        """
        interaction_matrix = interaction_matrix.tolil()
        target_matrix = target_matrix.tolil()
        minimized_matrix = lil_matrix(interaction_matrix.shape, dtype=interaction_matrix.dtype)

        for user_idx, (interaction_vector, target_vector) in enumerate(zip(interaction_matrix, target_matrix)):
            self.result_builder.start_timer(user_ids=user_idx)
            self.result_builder.set_sample_counts(user_ids=user_idx, counts=0)

            # Contains the candidates and selected interactions for the current user
            minimized_vector = lil_matrix((1, interaction_matrix.shape[1]), dtype=interaction_matrix.dtype)
            candidates_matrix = lil_matrix((1, interaction_matrix.shape[1]), dtype=interaction_matrix.dtype)

            map_candidate_to_item = np.array(range(interaction_vector.getnnz()), dtype=int)

            # Faster to pre-calculate and slice later
            repeated_interaction_matrix = repeat_lil_vector(vector=interaction_vector, times=interaction_vector.getnnz()).tocsr()
            repeated_target_matrix = repeat_lil_vector(vector=target_vector, times=interaction_vector.getnnz()).tocsr()

            if self.max_size is not None:
                max_size = min(self.max_size, interaction_vector.getnnz())

            while True:
                sliced_target_matrix = repeated_target_matrix[0 : candidates_matrix.shape[0], :]
                sliced_interaction_matrix = repeated_interaction_matrix[0 : candidates_matrix.shape[0], :]
                candidates_matrix = candidates_matrix.tocsr()

                # Forward pass: Predict and score
                scores = self._forward(candidates_matrix, sliced_interaction_matrix, sliced_target_matrix)
                self.result_builder.add_sample_count(
                    user_ids=user_idx, counts=candidates_matrix.shape[0]
                )

                best_candidate_index = np.argmax(scores)
                best_item_index = map_candidate_to_item[best_candidate_index]

                # Update selection and remove from possible candidates
                if candidates_matrix.nnz != 0:
                    minimized_vector[0, best_item_index] = 1
                    interaction_vector[0, best_item_index] = 0

                if self.max_size is not None:
                    # If max-size is specified, stop early
                    if minimized_vector.getnnz() == max_size:
                        self.result_builder.set_scores(
                            user_ids=user_idx, scores=scores[best_candidate_index]
                        )
                        break

                if scores[best_candidate_index] >= self.thresholds[user_idx]:
                    self.result_builder.set_scores(
                        user_ids=user_idx, scores=scores[best_candidate_index]
                    )
                    break

                # repeat the previously selected interactions
                candidates_matrix = repeat_lil_vector(minimized_vector, interaction_vector.getnnz())

                # add candidate interactions
                for i, item_idx in enumerate(interaction_vector.nonzero()[1]):
                    candidates_matrix[i, item_idx] = 1
                    map_candidate_to_item[i] = item_idx

            # minimized interactions for this user
            minimized_matrix[user_idx] = minimized_vector

            self.result_builder.stop_timer()
            if self.timeout_manager.has_timed_out():
                # If timeout, return the processed part of the matrix
                minimized_matrix = minimized_matrix[:(user_idx+1), :]
                break

        minimized_matrix = minimized_matrix.tocsr()
        self.result_builder.set_output_statistics(minimized_matrix)
        return self.result_builder