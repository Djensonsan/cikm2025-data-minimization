from recpack.matrix import InteractionMatrix
import pandas as pd
import pytest
import numpy as np

min_t = 0
max_t = 100

@pytest.fixture(scope="function")
def data_stratified():
    num_users = 100
    num_items = 100

    np.random.seed(42)

    user_ix, item_ix, timestamp_ix = [], [], []

    # Generate interactions such that each user has one more interaction than the previous
    for user in range(num_users):
        num_interactions_for_user = user + 2
        # Ensure we don't request more items than available
        num_interactions_for_user = min(num_interactions_for_user, num_items)
        # Generate unique item indices for each interaction
        unique_items = np.random.choice(range(num_items), size=num_interactions_for_user, replace=False)
        for item in unique_items:
            user_ix.append(user)
            item_ix.append(item)
            timestamp_ix.append(np.random.randint(min_t, max_t))

    input_dict = {
        InteractionMatrix.USER_IX: user_ix,
        InteractionMatrix.ITEM_IX: item_ix,
        InteractionMatrix.TIMESTAMP_IX: timestamp_ix,
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates([InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX], inplace=True)

    data = InteractionMatrix(
        df,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )
    return data



@pytest.fixture(scope="function")
def data():
    num_users = 50
    num_items = 100
    num_interactions = 5000

    np.random.seed(42)

    input_dict = {
        InteractionMatrix.USER_IX: [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        InteractionMatrix.ITEM_IX: [np.random.randint(0, num_items) for _ in range(0, num_interactions)],
        InteractionMatrix.TIMESTAMP_IX: [np.random.randint(min_t, max_t) for _ in range(0, num_interactions)],
    }

    df = pd.DataFrame.from_dict(input_dict)
    df.drop_duplicates([InteractionMatrix.USER_IX, InteractionMatrix.ITEM_IX], inplace=True)

    data = InteractionMatrix(
        df,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )
    return data