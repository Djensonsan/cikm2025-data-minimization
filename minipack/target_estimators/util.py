import numpy as np
from typing import Optional
from scipy.sparse import csr_matrix

def get_top_K_ranks(X: csr_matrix, K: Optional[int] = None) -> csr_matrix:
    """Returns a matrix of ranks assigned to the largest K values in X.

    Selects the K largest values for every row in X and assigns a rank to each.

    :param X: Matrix from which we will select K values in every row.
    :type X: csr_matrix
    :param K: Amount of values to select.
    :type K: int, optional
    :return: Matrix with K values per row.
    :rtype: csr_matrix
    """
    U, I, V = [], [], []
    for row_ix, (le, ri) in enumerate(zip(X.indptr[:-1], X.indptr[1:])):
        K_row_pick = min(K, ri - le) if K is not None else ri - le

        if K_row_pick != 0:

            top_k_row = X.indices[le + np.argpartition(X.data[le:ri], list(range(-K_row_pick, 0)))[-K_row_pick:]]

            for rank, col_ix in enumerate(reversed(top_k_row)):
                U.append(row_ix)
                I.append(col_ix)
                V.append(rank + 1)

    X_top_K = csr_matrix((V, (U, I)), shape=X.shape)

    return X_top_K