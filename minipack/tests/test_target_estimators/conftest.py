import pytest
from scipy.sparse import csr_matrix


@pytest.fixture(scope="function")
def data():
    values = [1] * 7
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    d = csr_matrix((values, (users, items)), shape=(3, 3))

    return d


@pytest.fixture(scope="function")
def Y_pred_binary_simple():
    """A binary prediction."""
    values = [1] * 4
    users = [0, 1, 2, 2]
    items = [1, 0, 0, 1]
    d = csr_matrix((values, (users, items)), shape=(3, 3))

    return d


@pytest.fixture(scope="function")
def Y_true_binary_simple():
    """A binary ground truth."""
    values = [1] * 3
    users = [0, 1, 2]
    items = [2, 2, 2]
    d = csr_matrix((values, (users, items)), shape=(3, 3))

    return d


@pytest.fixture(scope="function")
def Y_true_binary():
    """A binary ground truth."""
    true_users, true_items = [0, 0, 1, 1, 1], [0, 2, 0, 1, 3]

    true_data = csr_matrix(
        ([1 for i in range(len(true_users))], (true_users, true_items)), shape=(2, 5)
    )

    return true_data


@pytest.fixture(scope="function")
def Y_true_binary_simplified():
    """A simplified binary ground truth."""
    true_users, true_items = [0, 1], [2, 4]

    true_data = csr_matrix(
        ([1 for i in range(len(true_users))], (true_users, true_items)), shape=(2, 5)
    )

    return true_data

@pytest.fixture(scope="function")
def Y_pred():
    """A non-binary prediction."""
    pred_users, pred_items, pred_values = (
        [0, 0, 0, 1, 1, 1],
        [0, 2, 3, 1, 3, 4],
        [0.4, 0.3, 0.2, 0.2, 0.4, 0.5],
    )

    pred = csr_matrix(
        (pred_values, (pred_users, pred_items)), shape=(2, 5)
    )

    return pred

@pytest.fixture(scope="function")
def Y_pred_switched():
    """Similar to Y_pred, but with two switched values."""
    pred_users, pred_items, pred_values = (
        [0, 0, 0, 1, 1, 1],
        [0, 2, 3, 1, 3, 4],
        [0.4, 0.3, 0.2, 0.4, 0.2, 0.5],
    )

    pred = csr_matrix(
        (pred_values, (pred_users, pred_items)), shape=(2, 5)
    )

    return pred
