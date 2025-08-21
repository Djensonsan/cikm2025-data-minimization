import logging
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from minipack.minimizers.base import BaseMinimizer
from minipack.minimizers.result import MinimizationResultBuilder

logger = logging.getLogger("minipack")


class RandomSelectionMinimizer(BaseMinimizer):
    """A minimizer that employs a random selection strategy.

    This class implements a minimization algorithm by randomly selecting interactions
    to include in the minimized dataset. The algorithm iteratively adds interactions
    until a specified performance criterion is met.

    The class inherits from BaseMinimizer and implements the `minimize` method.
    """

    def _setup(self, interaction_matrix: csr_matrix) -> tuple:
        """
        Prepares the necessary matrices and variables for minimization.
        (Added tracking for original matrix and failed/exhausted users)
        """
        minimized_matrix = csr_matrix(
            interaction_matrix.shape, dtype=interaction_matrix.dtype
        )
        # a map from new row indices (index) to original indices (value)
        lookup_original_user_idx = np.array(
            range(interaction_matrix.shape[0]), dtype=int
        )
        # holds rows that don't meet the performance threshold (original indices)
        users_below_threshold = np.array(range(interaction_matrix.shape[0]), dtype=int)

        original_interaction_matrix_lil = interaction_matrix.tolil()
        permanently_failed_users = set()
        exhausted_last_round_users = set()  # Track who was exhausted previous round

        return (
            minimized_matrix,
            lookup_original_user_idx,
            users_below_threshold,
            original_interaction_matrix_lil,  # Added
            permanently_failed_users,  # Added
            exhausted_last_round_users  # Added
        )

    def _minimize(
            self, interaction_matrix: csr_matrix, target_matrix: csr_matrix
    ) -> MinimizationResultBuilder:
        """Executes the minimization process using a random selection strategy.
        (Corrected filtering and exhaustion logic timing)
        """
        (
            minimized_matrix,
            lookup_original_user_idx,
            users_below_threshold,
            original_interaction_matrix_lil,  # Added
            permanently_failed_users,  # Added
            exhausted_last_round_users  # Added
        ) = self._setup(interaction_matrix)

        n_samples_per_row = 1
        while True:
            active_users_mask = ~np.isin(users_below_threshold, list(permanently_failed_users))
            active_users_indices = users_below_threshold[active_users_mask]
            active_lookup_original_user_idx = lookup_original_user_idx[active_users_mask]

            if active_users_indices.size == 0:
                logger.info("No active users remaining below threshold. Stopping.")
                if self.result_builder._timers:
                    self.result_builder.stop_timer(increment=False)
                break

            self.result_builder.start_timer(
                user_ids=active_users_indices, init_value=0.0
            )

            minimized_matrix_csr = minimized_matrix.tocsr()  # Ensure CSR for scoring
            scores = self._forward(
                input_matrix=minimized_matrix_csr[active_users_indices],
                history_matrix=interaction_matrix[active_users_indices],
                target_matrix=target_matrix[active_users_indices],
            )
            self.result_builder.add_sample_count(
                user_ids=active_users_indices, counts=1, init_value=0.0
            )
            self.result_builder.set_scores(
                user_ids=active_users_indices, scores=scores
            )

            relative_indices_still_below = np.where(
                scores < self.thresholds[active_users_indices]
            )[0]
            original_indices_still_below = active_lookup_original_user_idx[relative_indices_still_below]
            scores_of_still_below = scores[relative_indices_still_below]  # Keep scores for logging

            # These are users who were exhausted last round AND are still below threshold now
            newly_failed_users = []
            for i, user_idx in enumerate(original_indices_still_below):
                if user_idx in exhausted_last_round_users:
                    # This user failed even after full history was scored
                    if user_idx not in permanently_failed_users:
                        current_score = scores_of_still_below[i]
                        current_threshold = self.thresholds[user_idx]
                        logger.warning(
                            f"User {user_idx}: Confirmed failure. Score with full history "
                            f"({current_score:.4f}) is still below threshold ({current_threshold:.4f}). "
                            f"Excluding permanently."
                        )
                        permanently_failed_users.add(user_idx)
                        newly_failed_users.append(user_idx)  # Track to remove from sampling list

            # Remove newly failed users from those still below threshold
            users_for_sampling_this_round = np.setdiff1d(
                original_indices_still_below, newly_failed_users, assume_unique=True
            )

            # Update main lists for the next iteration's start
            users_below_threshold = users_for_sampling_this_round
            lookup_original_user_idx = users_below_threshold  # Update lookup to match

            if users_for_sampling_this_round.size == 0:
                # Stop if no one is left to sample (everyone met threshold or failed permanently)
                self.result_builder.stop_timer(increment=True)
                logger.info("All users either met threshold or failed permanently. Stopping.")
                break

            minimized_matrix_lil = minimized_matrix_csr.tolil()  # Convert back to LIL for updates
            exhausted_this_round = set()  # Reset for current sampling round

            for user_idx_new in range(users_for_sampling_this_round.size):
                user_idx_original = users_for_sampling_this_round[user_idx_new]

                item_indices = original_interaction_matrix_lil.rows[user_idx_original]
                values = original_interaction_matrix_lil.data[user_idx_original]
                num_available_items = len(item_indices)
                num_samples_to_take = min(n_samples_per_row, num_available_items)
                is_exhausted = (num_samples_to_take == num_available_items and num_available_items > 0)

                if num_samples_to_take > 0:
                    # Sampling logic...
                    sampled_item_locs = np.random.choice(
                        num_available_items, size=num_samples_to_take, replace=False
                    )
                    sampled_indices = [item_indices[idx] for idx in sampled_item_locs]
                    sampled_values = [values[idx] for idx in sampled_item_locs]
                    minimized_matrix_lil.rows[user_idx_original] = sampled_indices
                    minimized_matrix_lil.data[user_idx_original] = sampled_values
                else:
                    # Logging for no samples taken...
                    # (Retrieve score/threshold if needed - score array needs careful indexing)
                    logger.debug(f"No samples taken for user {user_idx_original}...")

                # If exhausted THIS round, track for NEXT iteration's check
                if is_exhausted:
                    exhausted_this_round.add(user_idx_original)

            minimized_matrix = minimized_matrix_lil.tocsr()  # Finalize matrix state for this round
            exhausted_last_round_users = exhausted_this_round  # Carry over exhaustion state
            n_samples_per_row += 1

            self.result_builder.stop_timer(increment=True)

            if self.timeout_manager.has_timed_out():
                logger.warning("Minimization stopped due to timeout.")
                break

        self.result_builder.set_output_statistics(minimized_matrix)
        return self.result_builder