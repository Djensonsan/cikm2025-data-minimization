import numpy as np
from scipy.sparse import csr_matrix
from recpack.metrics.base import ListwiseMetricK
from recpack.metrics.util import sparse_divide_nonzero
from minipack.target_estimators import BaseTargetEstimator, ExponentialDecayEstimator
from minipack.target_estimators.util import get_top_K_ranks

def binarize_csr_matrix(X: csr_matrix) -> csr_matrix:
    """
    Binarizes a CSR matrix, setting all non-zero values to 1.

    Args:
        X (csr_matrix): Input sparse matrix in CSR format.

    Returns:
        csr_matrix: Binarized copy of the input matrix.

    Note:
        Does not modify the original matrix.
    """
    # Create a copy of the matrix to avoid modifying the original data
    X_binary = X.copy()
    # Set all non-zero values to 1
    X_binary.data = np.ones_like(X_binary.data)
    return X_binary


class NDCG(ListwiseMetricK):
    """
    Calculates normalized Discounted Cumulative Gain (nDCG) for recommendation lists.

    nDCG measures the performance of a recommendation system based on the graded relevance
    of the recommended items. It normalizes the Discounted Cumulative Gain (DCG) by the Ideal
    DCG (IDCG) to ensure scores are between 0 and 1.

    Attributes:
        K (int): Number of top recommendations to consider.
        IDCG_cache (array, optional): Precomputed IDCG values to speed up calculation.

    Note:
        Requires K to be a positive integer and `ground_truth_template` to be callable if provided.
    """

    def __init__(self, K, ground_truth_template=None):
        if not isinstance(K, int):
            raise ValueError(f"K must be an integer, not {type(K)}")
        if not isinstance(ground_truth_template, BaseTargetEstimator) and ground_truth_template is not None:
            raise ValueError("ground_truth_template must be an instance of BaseTargetEstimator")
        if K <= 0:
            raise ValueError(f"K must be positive, not {K}")

        self.K = K
        self.IDCG_cache = None
        # In order to speed-up NDCG calculation, we can pre-compute IDCG values and save them in a cache
        # We can do this because both ground truth and discount are parametrized by rank
        if ground_truth_template is not None:
            ranks = np.arange(1, self.K + 1, dtype=np.float64)
            # A discount template to calculate discounted relevance scores:
            #   This is your standard choice of denominator for DCG
            discount_template = 1.0 / np.log2(ranks + 1)
            # Calculate relevance scores for each rank
            relevance_scores = ground_truth_template._estimator_function(ranks)
            # Calculate discounted relevance scores
            discounted_relevance = relevance_scores * discount_template
            # Calculate IDCG values by creating a list of partial sums (the
            # functional way)
            self.IDCG_cache = np.cumsum(discounted_relevance)
            self.IDCG_cache = np.concatenate(([1], self.IDCG_cache))

    def calculate(self, Y_true: csr_matrix, Y_pred: csr_matrix) -> csr_matrix:
        """
        Calculates the NDCG score at K for each user based on predictions and true values.

        Args:
            Y_true (csr_matrix): True relevance scores, shape [n_users, n_items].
            Y_pred (csr_matrix): Predicted relevance scores, shape [n_users, n_items].

        Returns:
            None
        """
        # Perform checks and cleaning
        # Y_true, Y_pred = self._eliminate_empty_users(Y_true, Y_pred)
        # self._verify_shape(Y_true, Y_pred)
        # self._set_shape(Y_true)

        self.scores_ = self._calculate(Y_true, Y_pred)

        return self.scores_

    # def calculate_lean(self, Y_true: csr_matrix, Y_pred: csr_matrix) -> csr_matrix:
    #     """
    #     Calculates the NDCG score at K for each user based on predictions and true values.
    #
    #     Differs from the "calculate" method in that it does not perform input checking or cleaning.
    #
    #     Args:
    #         Y_true (csr_matrix): True relevance scores, shape [n_users, n_items].
    #         Y_pred (csr_matrix): Predicted relevance scores, shape [n_users, n_items].
    #
    #     Returns:
    #         csr_matrix: NDCG scores for each user.
    #     """
    #     return self._calculate(Y_true, Y_pred)

    def _calculate(self, Y_true: csr_matrix, Y_pred: csr_matrix) -> csr_matrix:
        if self.IDCG_cache is not None:
            return self._NDCG_cached(Y_true, Y_pred)
        return self._NDCG_naive(Y_true, Y_pred)

    def _NDCG_cached(self, Y_true: csr_matrix, Y_pred: csr_matrix) -> csr_matrix:
        """
        Calculates NDCG using precomputed IDCG values for efficiency.

        This method is preferred when IDCG values are cached to avoid recomputing
        them for each call, speeding up the NDCG calculation process.

        Args:
            Y_true (csr_matrix): True relevance scores, shape [n_users, n_items].
            Y_pred (csr_matrix): Predicted relevance scores, shape [n_users, n_items].

        Returns:
            csr_matrix: NDCG scores for each user, utilizing cached IDCG values.
        """
        Y_true_bin = binarize_csr_matrix(Y_true)

        Y_pred_top_K = get_top_K_ranks(Y_pred, self.K)

        # Calculate DCG
        # Denominator: log2(1 + rank)
        denominator = Y_pred_top_K.multiply(Y_true_bin)
        denominator.data = np.log2(denominator.data + 1)

        # Numerator: relevance scores of Y
        numerator = Y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        per_user_dcg = dcg.sum(axis=1)

        hist_len = Y_true_bin.sum(axis=1).astype(np.int32)
        hist_len[hist_len > self.K] = self.K

        self.scores_ = sparse_divide_nonzero(
            csr_matrix(per_user_dcg),
            csr_matrix(self.IDCG_cache[hist_len]),
        )

        return self.scores_

    def _NDCG_naive(self, Y_true: csr_matrix, Y_pred: csr_matrix) -> csr_matrix:
        """
        Calculates NDCG without using precomputed IDCG values, performing a naive computation.

        This method computes IDCG values on-the-fly for each calculation, making it
        slower than the cached version but more straightforward.

        Args:
            Y_true (csr_matrix): True relevance scores, shape [n_users, n_items].
            Y_pred (csr_matrix): Predicted relevance scores, shape [n_users, n_items].

        Returns:
            csr_matrix: NDCG scores for each user, calculated naively without using cached IDCG.
        """
        # Binary version of Y:
        Y_true_bin = binarize_csr_matrix(Y_true)

        Y_pred_top_K = get_top_K_ranks(Y_pred, self.K)

        # Calculate DCG
        # Denominator: log2(1 + rank)
        denominator = Y_pred_top_K.multiply(Y_true_bin)
        denominator.data = np.log2(denominator.data + 1)

        # Numerator: relevance scores of Y
        numerator = Y_true

        dcg = sparse_divide_nonzero(numerator, denominator)

        per_user_dcg = dcg.sum(axis=1)

        # Calculate IDCG
        Y_true_top_K = get_top_K_ranks(Y_true, self.K)
        # Denominator: log2(1 + rank)
        denominator = Y_true_top_K
        denominator.data = np.log2(denominator.data + 1)

        idcg = sparse_divide_nonzero(numerator, denominator)
        per_user_idcg = idcg.sum(axis=1)

        self.scores_ = sparse_divide_nonzero(
            csr_matrix(per_user_dcg),
            csr_matrix(per_user_idcg),
        )

        return self.scores_
