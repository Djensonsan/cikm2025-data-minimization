import pytest
from scipy.sparse import csr_matrix

@pytest.fixture(scope="function")
def simple_train_matrix():
    """
    The similarity matrix (S) of ItemKNN when you fit on 'data' is:
    [0.00, 0.50, 0.81]
    [0.50, 0.00, 0.81]
    [0.81, 0.81, 0.00]
    """
    values = [1] * 7
    users = [0, 0, 1, 1, 2, 2, 2]
    items = [1, 2, 0, 2, 0, 1, 2]
    d = csr_matrix((values, (users, items)), shape=(3, 3))

    return d

@pytest.fixture(scope="function")
def simple_interaction_matrix():
    """
    Our input history (X) is:
    [0.00, 1.00, 0.00]
    [1.00, 0.00, 0.00]
    [1.00, 1.00, 0.00]
    """
    values = [1] * 4
    users = [0, 1, 2, 2]
    items = [1, 0, 0, 1]
    d = csr_matrix((values, (users, items)), shape=(3, 3))

    return d


@pytest.fixture(scope="function")
def simple_target_matrix():
    """"
    Our ground truth (Y) is:
    [0.00, 0.00, 1.00]
    [0.00, 0.00, 1.00]
    [0.00, 0.00, 1.00]
    """
    values = [1] * 3
    users = [0, 1, 2]
    items = [2, 2, 2]
    d = csr_matrix((values, (users, items)), shape=(3, 3))

    return d

@pytest.fixture(scope="function")
def X_pred():
    """An input dataset."""
    pred_users, pred_items, pred_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [0.4, 0.3, 0.2, 0.2, 0.4, 0.5],
    )

    pred = csr_matrix(
        (pred_values, (pred_users, pred_items)), shape=(10, 5)
    )

    return pred


@pytest.fixture(scope="function")
def Y_true():
    """A binary ground truth."""
    true_users, true_items = [0, 0, 2, 2, 2], [0, 2, 0, 1, 3]

    true_data = csr_matrix(
        ([1 for i in range(len(true_users))], (true_users, true_items)), shape=(10, 5)
    )

    return true_data


@pytest.fixture(scope="function")
def Y_simplified():
    """A simplified binary ground truth.

    """
    true_users, true_items = [0, 2], [2, 4]

    true_data = csr_matrix(
        ([1 for i in range(len(true_users))], (true_users, true_items)), shape=(10, 5)
    )

    return true_data

@pytest.fixture(scope="function")
def Y_predictions():
    """An input dataset."""
    pred_users, pred_items, pred_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [0.3, 0.2, 0.1, 0.3, 0.23, 0.5],
    )

    pred = csr_matrix(
        (pred_values, (pred_users, pred_items)), shape=(10, 5)
    )

    return pred
