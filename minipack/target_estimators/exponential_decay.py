from scipy.sparse import csr_matrix
from minipack.target_estimators.base import BaseTargetEstimator
from minipack.target_estimators.util import get_top_K_ranks

class ExponentialDecayEstimator(BaseTargetEstimator):
    def __init__(self, K: int, gamma: float):
        if K <= 0:
            raise ValueError("K must be greater than 0")
        if gamma == 0:
            raise ValueError("Gamma must be different from 0")
        self.K = K
        self.gamma = gamma

    def estimate(self, Y: csr_matrix) -> csr_matrix:
        Y = get_top_K_ranks(Y, self.K)
        Y.data = self._estimator_function(Y.data)
        return Y

    def _estimator_function(self, rank):
        num = (rank + 1) ** self.gamma - (self.K + 1) ** self.gamma
        den = 2 ** self.gamma - (self.K + 1) ** self.gamma
        return num / den