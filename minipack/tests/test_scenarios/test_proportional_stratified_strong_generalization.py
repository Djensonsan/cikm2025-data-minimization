import pytest
import numpy as np
from minipack.scenarios.proportional_stratified_strong_generalization import ProportionalStratifiedStrongGeneralization

@pytest.mark.parametrize(
    "num_users_test, num_users_val, num_users_est",
    [(10, 10, 10), (20, 20, 20), (30, 10, 10)],
)
def test_stratified_strong_generalization_split(
    data_stratified,
    num_users_test,
    num_users_val,
    num_users_est,
    bin_width=10,
    bin_range=(0, 100),
    frac_interactions_in=0.5,
):
    scenario = ProportionalStratifiedStrongGeneralization(
        num_users_test=num_users_test,
        num_users_val=num_users_val,
        num_users_est=num_users_est,
        bin_width=bin_width,
        bin_range=bin_range,
        frac_interactions_in=frac_interactions_in,
        validation=False,
        estimation=False,
    )
    scenario.split(data_stratified)

    full_train = scenario.full_training_data

    test_data_in = scenario.test_data_in
    test_data_out = scenario.test_data_out

    test_data_in_bins = scenario.test_data_in_bins
    test_data_out_bins = scenario.test_data_out_bins

    assert test_data_in.num_active_users == num_users_test
    assert test_data_in.num_active_users == test_data_out.num_active_users

    full_train_users = set(full_train.indices[0])

    assert not full_train_users.intersection(test_data_in.indices[0])
    assert not full_train_users.intersection(test_data_out.indices[0])

    num_bins = (bin_range[1] - bin_range[0]) / bin_width

    assert len(test_data_in_bins) == num_bins

    for test_data_in_bin in test_data_in_bins:
        # no intersection between test and train users:
        assert not full_train_users.intersection(test_data_in_bin.indices[0])
        # bins must be subset of the entire test data
        assert set(test_data_in_bin.indices[0]) <= set(test_data_in.indices[0])

    for test_data_out_bin in test_data_out_bins:
        # no intersection between test and train users:
        assert not full_train_users.intersection(test_data_out_bin.indices[0])
        # bins must be subset of the entire test data
        assert set(test_data_out_bin.indices[0]) <= set(test_data_out.indices[0])

    # assert on the number of users in train
    assert (
        full_train.num_active_users == data_stratified.num_active_users - num_users_test
    )

    # assert on the number of interactions
    test_in_interactions = test_data_in.indices[1]
    test_out_interactions = test_data_out.indices[1]

    # Higher volatility, so not as bad to miss
    diff_allowed = 0.2
    assert (
        abs(len(test_in_interactions) / (len(test_in_interactions) + len(test_out_interactions)) - frac_interactions_in)
        < diff_allowed
    )

@pytest.mark.parametrize(
    "num_users_test, num_users_val, num_users_est",
    [(10, 10, 10), (20, 20, 20), (30, 10, 10)],
)
def test_stratified_strong_generalization_split_w_validation(
    data_stratified,
    num_users_test,
    num_users_val,
    num_users_est,
    bin_width=10,
    bin_range=(0, 100),
    frac_interactions_in=0.5,
):
    scenario = ProportionalStratifiedStrongGeneralization(
        num_users_test=num_users_test,
        num_users_val=num_users_val,
        num_users_est=num_users_est,
        bin_width=bin_width,
        bin_range=bin_range,
        frac_interactions_in=frac_interactions_in,
        validation=True,
        estimation=True,
    )
    scenario.split(data_stratified)

    full_train = scenario.full_training_data

    test_data_in = scenario.test_data_in
    test_data_out = scenario.test_data_out

    test_data_in_bins = scenario.test_data_in_bins
    test_data_out_bins = scenario.test_data_out_bins

    validation_train = scenario.validation_training_data
    validation_data_in = scenario.validation_data_in
    validation_data_out = scenario.validation_data_out

    estimation_data_in = scenario.estimation_data_in
    estimation_data_out = scenario.estimation_data_out

    estimation_data_in_bins = scenario.estimation_data_in_bins
    estimation_data_out_bins = scenario.estimation_data_out_bins

    # Assertions on the number of active users:
    assert test_data_in.num_active_users == num_users_test
    assert test_data_in.num_active_users == test_data_out.num_active_users

    assert validation_data_in.num_active_users == num_users_val
    assert validation_data_in.num_active_users == validation_data_out.num_active_users

    assert estimation_data_in.num_active_users == num_users_est
    assert estimation_data_in.num_active_users == estimation_data_out.num_active_users

    assert (
        validation_train.num_active_users
        == data_stratified.num_active_users
        - num_users_test
        - num_users_val
        - num_users_est
    )
    assert (
        full_train.num_active_users == data_stratified.num_active_users - num_users_test
    )

    # Assertions on the sets of active users:
    full_train_users = set(full_train.indices[0])
    test_in_users = set(test_data_in.indices[0])
    test_out_users = set(test_data_out.indices[0])

    assert not full_train_users.intersection(test_in_users)
    assert not full_train_users.intersection(test_out_users)

    val_train_users = set(validation_train.indices[0])
    val_in_users = set(validation_data_in.indices[0])
    val_out_users = set(validation_data_out.indices[0])

    est_in_users = set(estimation_data_in.indices[0])
    est_out_users = set(estimation_data_out.indices[0])

    assert not test_in_users.intersection(val_in_users)
    assert not test_out_users.intersection(val_out_users)

    assert not test_in_users.intersection(est_in_users)
    assert not test_out_users.intersection(est_out_users)

    assert not val_train_users.intersection(val_in_users)
    assert not val_train_users.intersection(val_out_users)

    assert not val_train_users.intersection(est_in_users)
    assert not val_train_users.intersection(est_out_users)

    assert not val_in_users.intersection(est_in_users)
    assert not val_out_users.intersection(est_out_users)

    assert val_in_users == val_out_users
    assert est_in_users == est_out_users

    num_bins = (bin_range[1] - bin_range[0]) / bin_width

    assert len(test_data_in_bins) == num_bins
    assert len(test_data_out_bins) == num_bins
    assert len(estimation_data_in_bins) == num_bins
    assert len(estimation_data_out_bins) == num_bins

    for est_data_in_bin in estimation_data_in_bins:
        assert not val_train_users.intersection(est_data_in_bin.indices[0])
        assert not val_in_users.intersection(est_data_in_bin.indices[0])
        # bins must be subset of the entire est data
        assert set(est_data_in_bin.indices[0]) <= set(estimation_data_in.indices[0])

    for est_data_out_bin in estimation_data_out_bins:
        assert not val_train_users.intersection(est_data_out_bin.indices[0])
        assert not val_out_users.intersection(est_data_out_bin.indices[0])
        # bins must be subset of the entire est data
        assert set(est_data_out_bin.indices[0]) <= set(estimation_data_out.indices[0])


def test_stratified_strong_generalization_invalid_number_users(data):
    num_users_test = -10
    num_users_val = 10
    num_users_est = 10
    bin_width = 10
    bin_range = (0, 100)
    frac_interactions_in = 0.5

    with pytest.raises(ValueError):
        scenario = ProportionalStratifiedStrongGeneralization(
            num_users_test=num_users_test,
            num_users_val=num_users_val,
            num_users_est=num_users_est,
            bin_width=bin_width,
            bin_range=bin_range,
            frac_interactions_in=frac_interactions_in,
            validation=True,
        )
        scenario.split(data)


def test_stratified_strong_generalization_invalid_negative_bin_width(data_stratified):
    # bin_width should be positive:
    num_users_test = 10
    num_users_val = 10
    num_users_est = 10
    bin_width = -1
    bin_range = (0, 100)
    frac_interactions_in = 0.5

    with pytest.raises(ValueError):
        scenario = ProportionalStratifiedStrongGeneralization(
            num_users_test=num_users_test,
            num_users_val=num_users_val,
            num_users_est=num_users_est,
            bin_width=bin_width,
            bin_range=bin_range,
            frac_interactions_in=frac_interactions_in,
            validation=True,
        )
        scenario.split(data_stratified)


def test_stratified_strong_generalization_invalid_size_bin_width(data_stratified):
    # bin_width should be smaller than second element of bin_range:
    num_users_test = 10
    num_users_val = 10
    num_users_est = 10
    bin_width = 20
    bin_range = (0, 10)
    frac_interactions_in = 0.5

    with pytest.raises(ValueError):
        scenario = ProportionalStratifiedStrongGeneralization(
            num_users_test=num_users_test,
            num_users_val=num_users_val,
            num_users_est=num_users_est,
            bin_width=bin_width,
            bin_range=bin_range,
            frac_interactions_in=frac_interactions_in,
            validation=True,
        )
        scenario.split(data_stratified)


def test_stratified_strong_generalization_invalid_order_bin_range(data_stratified):
    num_users_test = 10
    num_users_val = 10
    num_users_est = 10
    bin_width = 1
    # bin_range should be in increasing order:
    bin_range = (10, 1)
    frac_interactions_in = 0.5

    with pytest.raises(ValueError):
        scenario = ProportionalStratifiedStrongGeneralization(
            num_users_test=num_users_test,
            num_users_val=num_users_val,
            num_users_est=num_users_est,
            bin_width=bin_width,
            bin_range=bin_range,
            frac_interactions_in=frac_interactions_in,
            validation=True,
        )
        scenario.split(data_stratified)


def test_stratified_strong_generalization_invalid_negative_bin_range(data_stratified):
    # bin_range can't have negative elements:
    num_users_test = 10
    num_users_val = 10
    num_users_est = 10
    bin_width = 1
    bin_range = (-1, 10)
    frac_interactions_in = 0.5

    with pytest.raises(ValueError):
        scenario = ProportionalStratifiedStrongGeneralization(
            num_users_test=num_users_test,
            num_users_val=num_users_val,
            num_users_est=num_users_est,
            bin_width=bin_width,
            bin_range=bin_range,
            frac_interactions_in=frac_interactions_in,
            validation=True,
        )
        scenario.split(data_stratified)


def test_stratified_strong_generalization_invalid_bin_range_non_division(
    data_stratified,
):
    # bin_range can't have negative elements:
    num_users_test = 10
    num_users_val = 10
    num_users_est = 10
    bin_width = 3
    bin_range = (0, 10)
    frac_interactions_in = 0.5

    with pytest.raises(ValueError):
        scenario = ProportionalStratifiedStrongGeneralization(
            num_users_test=num_users_test,
            num_users_val=num_users_val,
            num_users_est=num_users_est,
            bin_width=bin_width,
            bin_range=bin_range,
            frac_interactions_in=frac_interactions_in,
            validation=True,
        )
        scenario.split(data_stratified)


@pytest.mark.parametrize(
    "num_users_test, num_users_val, num_users_est",
    [(10, 10, 10), (20, 20, 20), (30, 10, 10)],
)
def test_stratified_strong_generalization_same_seed(
    data_stratified,
    num_users_test,
    num_users_val,
    num_users_est,
    bin_width=10,
    bin_range=(0, 100),
    frac_interactions_in=0.5,
):
    # First scenario uses a random seed
    scenario_1 = ProportionalStratifiedStrongGeneralization(
        num_users_test=num_users_test,
        num_users_val=num_users_val,
        num_users_est=num_users_est,
        bin_width=bin_width,
        bin_range=bin_range,
        frac_interactions_in=frac_interactions_in,
        validation=True,
    )
    seed = scenario_1.seed
    scenario_1.split(data_stratified)

    # second scenario uses same seed as the previous one
    scenario_2 = ProportionalStratifiedStrongGeneralization(
        num_users_test=num_users_test,
        num_users_val=num_users_val,
        num_users_est=num_users_est,
        bin_width=bin_width,
        bin_range=bin_range,
        frac_interactions_in=frac_interactions_in,
        validation=True,
        seed=seed,
    )
    scenario_2.split(data_stratified)

    assert (
        scenario_1.full_training_data.num_interactions
        == scenario_2.full_training_data.num_interactions
    )
    for index, test_data_bin in enumerate(scenario_1.test_data):
        assert (
            test_data_bin.num_interactions
            == scenario_2.test_data[index].num_interactions
        )

@pytest.mark.parametrize(
    "num_users_test, num_users_val, num_users_est",
    [(10, 10, 10), (20, 20, 20), (30, 10, 10)],
)
def test_stratified_strong_generalization_different_seeds(
    data_stratified,
    num_users_test,
    num_users_val,
    num_users_est,
    bin_width=10,
    bin_range=(0, 100),
    frac_interactions_in=0.5,
):
    # First scenario with a specific seed
    scenario_1 = ProportionalStratifiedStrongGeneralization(
        num_users_test=num_users_test,
        num_users_val=num_users_val,
        num_users_est=num_users_est,
        bin_width=bin_width,
        bin_range=bin_range,
        frac_interactions_in=frac_interactions_in,
        validation=True,
        seed=42,  # Set a specific seed
    )
    scenario_1.split(data_stratified)

    # Second scenario with a different seed
    scenario_2 = ProportionalStratifiedStrongGeneralization(
        num_users_test=num_users_test,
        num_users_val=num_users_val,
        num_users_est=num_users_est,
        bin_width=bin_width,
        bin_range=bin_range,
        frac_interactions_in=frac_interactions_in,
        validation=True,
        seed=43,  # Set a different seed
    )
    scenario_2.split(data_stratified)

    # Convert CSR matrices to dense NumPy arrays for comparison
    full_train_1 = scenario_1.full_training_data.binary_values.toarray()
    full_train_2 = scenario_2.full_training_data.binary_values.toarray()

    # Check that the full training data is different
    assert not np.array_equal(full_train_1, full_train_2), \
        "The full training data should differ with different seeds."
