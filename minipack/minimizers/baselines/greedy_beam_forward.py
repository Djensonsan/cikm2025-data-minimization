import logging
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, vstack
from recpack.algorithms import Algorithm
from recpack.metrics.base import Metric
from minipack.minimizers.base import BaseMinimizer
from minipack.minimizers.util import repeat_lil_vector, deduplicate_lil
from minipack.minimizers.result import MinimizationResultBuilder

logger = logging.getLogger("minipack")


class GreedyBeamForwardMinimizer(BaseMinimizer):
    """
    A minimizer that employs a greedy beam forward selection strategy.

    This class extends the BaseMinimizer to implement a minimization algorithm using a beam search approach.
    It iteratively selects interactions to include in the minimized dataset, aiming to optimize a performance
    metric at each step. The selection process is guided by a "beam" of candidate solutions, with the beam width
    (beam_depth) controlling how many candidates are considered simultaneously.

    The algorithm enhances the greedy forward selection strategy by considering multiple potential paths through the
    solution space, thereby potentially increasing the quality of the final minimized dataset by balancing between
    exploration and exploitation.
    """
    def __init__(
        self,
        metric: Metric = None,
        model: Algorithm = None,
        eta: float = 0.9,
        remove_history: bool = True,
        beam_depth=5,
        timeout=float("inf"),
        timeout_manager=None,
    ):
        """
        Initializes a GreedyBeamForwardMinimizer instance with specified metric, model, performance threshold,
        and beam depth.

        Args:
            metric (callable, optional): A function that computes the metric to be used for minimization. This could be any callable
                that takes as input the predictions and true labels and returns a float representing the metric value.
                Defaults to None.
            model (Algorithm, optional): The algorithm instance adhering to the RecPack Algorithm interface,
                used here for making inferences during the minimization process. Defaults to None.
            performance_threshold (np.ndarray or torch.Tensor, optional): Threshold values that specifies the acceptable level of performance.
                The minimizer will aim to ensure that the performance of the minimized dataset is above this threshold.
                Defaults to None.
            beam_depth (int, optional): The beam width for the beam search strategy, defining how many candidate
                                        solutions are evaluated in parallel at each step. Defaults to 5.
            timeout (int, optional): The maximum time in seconds allowed for the minimization process. Defaults to 600.

        Raises:
            ValueError: If `beam_depth` is not a positive integer.
        """
        super().__init__(metric, model, eta, remove_history, timeout, timeout_manager)

        if not isinstance(beam_depth, int) or beam_depth <= 0:
            raise ValueError("beam_depth must be a positive integer")

        self.beam_depth = beam_depth

    def _minimize(
        self, interaction_matrix: csr_matrix, target_matrix: csr_matrix
    ) -> MinimizationResultBuilder:
        """Executes the minimization process using a greedy beam forward selection strategy.

        Args:
            interaction_matrix (csr_matrix): A sparse matrix representing input data for the model.
            target_matrix (csr_matrix): A sparse matrix representing ground truth data.

        Returns:
            MinimizationResult: An object containing the results of the minimization process.
        """
        interaction_matrix = interaction_matrix.tolil()
        target_matrix = target_matrix.tolil()
        minimized_matrix = lil_matrix(
            interaction_matrix.shape, dtype=interaction_matrix.dtype
        )

        for user_idx, (interaction_vector, target_vector) in enumerate(
            zip(interaction_matrix, target_matrix)
        ):
            self.result_builder.start_timer(user_ids=user_idx)
            self.result_builder.set_sample_counts(user_ids=user_idx, counts=0)

            candidates_matrix = lil_matrix(
                (1, interaction_matrix.shape[1]), dtype=interaction_matrix.dtype
            )

            # Faster to pre-calculate and slice later
            repeated_interaction_matrix = repeat_lil_vector(
                vector=interaction_vector,
                times=interaction_vector.getnnz() * self.beam_depth,
            ).tocsr()
            repeated_target_matrix = repeat_lil_vector(
                vector=target_vector,
                times=interaction_vector.getnnz() * self.beam_depth,
            ).tocsr()

            while True:
                # Forward pass: Predict and score
                sliced_target_matrix = repeated_target_matrix[
                    0 : candidates_matrix.shape[0], :
                ]
                sliced_interaction_matrix = repeated_interaction_matrix[
                    0 : candidates_matrix.shape[0], :
                ]
                candidates_matrix = candidates_matrix.tocsr()

                scores = self._forward(
                    input_matrix=candidates_matrix,
                    history_matrix=sliced_interaction_matrix,
                    target_matrix=sliced_target_matrix,
                )
                self.result_builder.add_sample_count(
                    user_ids=user_idx, counts=candidates_matrix.shape[0]
                )

                # Get best performing candidates
                beam_depth = min(self.beam_depth, len(scores))
                best_candidates_indices = np.argpartition(scores, -beam_depth)[
                    -beam_depth:
                ]
                best_candidates_indices = best_candidates_indices[
                    np.argsort(scores[best_candidates_indices])
                ]

                # If the performance is valid, we stop here and return the best performing candidate
                if scores[best_candidates_indices[-1]] >= self.thresholds[user_idx]:
                    minimized_vector = candidates_matrix[best_candidates_indices[-1]]
                    self.result_builder.set_scores(
                        user_ids=user_idx, scores=scores[best_candidates_indices[-1]]
                    )
                    break

                # Select the best performing candidates as the new beams
                beam_matrix = candidates_matrix[best_candidates_indices].tolil()

                # Contains all candidate interactions for the current beams
                candidates_matrix = lil_matrix(
                    (0, interaction_matrix.shape[1]), dtype=interaction_matrix.dtype
                )

                # Add candidate interactions for each beam
                for beam_vector in beam_matrix:
                    beam_candidate_set = set(interaction_vector.rows[0]) - set(
                        beam_vector.rows[0]
                    )
                    candidates_beam_matrix = repeat_lil_vector(
                        beam_vector, len(beam_candidate_set)
                    )
                    for i, item_idx in enumerate(beam_candidate_set):
                        candidates_beam_matrix[i, item_idx] = 1
                    candidates_matrix = vstack(
                        [candidates_matrix, candidates_beam_matrix], format="lil"
                    )

                # Remove duplicates
                candidates_matrix = deduplicate_lil(candidates_matrix)

            # Selected interactions for this user
            minimized_matrix[user_idx] = minimized_vector

            self.result_builder.stop_timer()
            if self.timeout_manager.has_timed_out():
                # If timeout, return the processed part of the matrix
                minimized_matrix = minimized_matrix[: (user_idx + 1), :]
                break

        minimized_matrix = minimized_matrix.tocsr()
        self.result_builder.set_output_statistics(minimized_matrix)
        return self.result_builder
