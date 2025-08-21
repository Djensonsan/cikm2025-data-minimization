import pytest

from minipack.scenarios.strong_generalization import StrongGeneralization

@pytest.mark.parametrize("num_users_test, num_users_val, frac_interactions_in", [(10, 10, 0.5), (5, 10, 0.5), (3, 9, 0.5)])
def test_strong_generalization_split(data, num_users_test, num_users_val, frac_interactions_in):
    scenario = StrongGeneralization(num_users_test, num_users_val, frac_interactions_in)
    scenario.split(data)

    full_train = scenario.full_training_data
    test_data_in, test_data_out = scenario.test_data

    test_in_users = set(test_data_in.indices[0])
    test_out_users = set(test_data_out.indices[0])
    full_train_users = set(full_train.indices[0])

    assert not full_train_users.intersection(test_in_users)
    assert not full_train_users.intersection(test_out_users)
    assert test_in_users == test_out_users
    assert test_data_in.num_active_users == num_users_test
    assert test_data_out.num_active_users == num_users_test


    test_in_interactions = test_data_in.indices[1]
    test_out_interactions = test_data_out.indices[1]

    diff_allowed = 0.2
    assert (
        abs(len(test_in_interactions) / (len(test_in_interactions) + len(test_out_interactions)) - frac_interactions_in)
        < diff_allowed
    )

@pytest.mark.parametrize("num_users_test, num_users_val, frac_interactions_in", [(10, 10, 0.5), (5, 10, 0.5), (3, 9, 0.5)])
def test_strong_generalization_split_w_validation(data, num_users_test, num_users_val, frac_interactions_in):
    scenario = StrongGeneralization(num_users_test, num_users_val, frac_interactions_in, validation=True)
    scenario.split(data)

    val_train = scenario.validation_training_data
    full_train = scenario.full_training_data
    val_data_in, val_data_out = scenario.validation_data
    test_data_in, test_data_out = scenario.test_data

    val_train_users = set(val_train.indices[0])
    full_train_users = set(full_train.indices[0])
    test_in_users = set(test_data_in.indices[0])
    test_out_users = set(test_data_out.indices[0])
    val_in_users = set(val_data_in.indices[0])
    val_out_users = set(val_data_out.indices[0])

    assert not val_train_users.intersection(test_in_users)
    assert not val_train_users.intersection(val_in_users)
    assert not test_in_users.intersection(val_in_users)
    assert not full_train_users.intersection(test_in_users)
    assert full_train_users.intersection(val_in_users) == val_in_users
    assert full_train_users.intersection(val_train_users) == val_train_users

    assert val_data_in.num_active_users > 0

    assert test_in_users == test_out_users
    assert val_in_users == val_out_users

    assert val_data_in.num_active_users == num_users_val
    assert val_data_out.num_active_users == num_users_val
    assert test_data_in.num_active_users == num_users_test
    assert test_data_out.num_active_users == num_users_test

    assert val_data_out.active_users == val_data_in.active_users
    assert test_data_out.active_users == test_data_in.active_users

@pytest.mark.parametrize("num_users_test, num_users_val", [(25, 25), (100, 10), (10, 100)])
def test_strong_generalization_split_w_validation_error(data, num_users_test, num_users_val):
    with pytest.raises(ValueError):
        scenario = StrongGeneralization(num_users_test, num_users_val, validation=True)
        scenario.split(data)

@pytest.mark.parametrize("num_users_test, num_users_val", [(10, 10), (5, 10), (3, 9)])
def test_strong_generalization_split_seed(data, num_users_test, num_users_val):
    # First scenario uses a random seed
    scenario_1 = StrongGeneralization(num_users_test, num_users_val, validation=True)
    seed = scenario_1.seed
    scenario_1.split(data)

    # second scenario uses same seed as the previous one
    scenario_2 = StrongGeneralization(num_users_test, num_users_val, validation=True, seed=seed)
    scenario_2.split(data)

    assert scenario_1.full_training_data.num_interactions == scenario_2.full_training_data.num_interactions
    assert scenario_1.test_data_in.num_interactions == scenario_2.test_data_in.num_interactions
    assert scenario_1.test_data_out.num_interactions == scenario_2.test_data_out.num_interactions