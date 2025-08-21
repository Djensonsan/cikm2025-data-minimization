import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class MinimizationResult:
    minimizer_identifier: str = None
    estimator_identifier: str = None
    batch_id: int = None
    minimized_matrix: csr_matrix = None
    batch_number_of_users: int = None
    batch_number_of_users_processed: int = None
    batch_percentage_of_users_processed: float = None
    batch_original_number_of_input_interactions: int = None
    batch_minimized_number_of_input_interactions: int = None
    batch_number_of_target_interactions: int = None
    batch_minimization_ratio: float = None
    batch_runtime: float = None
    batch_sample_count: int = None
    batch_constraint_satisfaction: float = None
    average_original_number_of_input_interactions: float = None
    average_minimized_number_of_input_interactions: float = None
    average_number_of_target_interactions: float = None
    average_runtime: float = None
    average_sample_count: float = None
    timeout_occurred: bool = None
    per_user_scores: np.ndarray = None
    per_user_original_input_interactions: np.ndarray = None
    per_user_minimized_input_interactions: np.ndarray = None
    per_user_target_interactions: np.ndarray = None
    per_user_minimization_ratios: np.ndarray = None
    per_user_runtimes: np.ndarray = None
    per_user_sample_counts: np.ndarray = None
    per_user_performance_threshold: np.ndarray = None
    per_user_constraint_satisfaction: np.ndarray = None
    additional_batch_metrics: Dict[str, Any] = field(default_factory=dict)
    additional_user_metrics: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def batch_results(self):
        attributes = [
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
        filtered_data = {key: getattr(self, key) for key in attributes}
        filtered_data.update(self.additional_batch_metrics)
        return pd.DataFrame([filtered_data])

    @property
    def user_results(self):
        attributes = [
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
        filtered_data = {key: getattr(self, key) for key in attributes}
        filtered_data.update(self.additional_user_metrics)

        # Number of users in the dataset
        num_users = len(self.per_user_scores)

        # Repeat the constant columns for each user
        constant_columns = {
            "minimizer_identifier": [self.minimizer_identifier] * num_users,
            "estimator_identifier": [self.estimator_identifier] * num_users,
            "batch_id": [self.batch_id] * num_users,
        }

        # Convert user-specific attributes to a DataFrame
        user_data = pd.DataFrame(filtered_data)

        # Add constant columns to user-specific data
        for key, value in constant_columns.items():
            user_data[key] = value

        # Reorder the columns so that constant columns come first
        ordered_columns = ["minimizer_identifier", "estimator_identifier", "batch_id"] + list(user_data.columns[:-3])
        user_data = user_data[ordered_columns]

        return user_data