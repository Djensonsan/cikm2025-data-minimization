import logging
import numpy as np
from warnings import warn
from typing import Tuple, Union, Dict, List, Optional, Any
from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.scenarios.splitters import Splitter, UserSplitter
from recpack.scenarios.splitters import FractionInteractionSplitter
from minipack.scenarios.splitters import (
    ProportionalStratifiedStrongGeneralizationSplitter,
    StrongGeneralizationSplitter,
)

logger = logging.getLogger("minipack")


class ProportionalStratifiedStrongGeneralization(Scenario):
    """Predict (randomly) held-out interactions of previously unseen users.

    During splitting each user is randomly assigned to one of four groups of users:
    training, validation, estimation and testing. This is a version of the
    StrongGeneralization scenario, where the users are returned as bins based on their
    number of interactions.

    Test, estimation and validation users' interactions are split into a `data_in` (fold-in) and
    `data_out` (held-out) set.

    ``frac_interactions_in`` of interactions are assigned to fold-in, the remainder
    to the held-out set.

    Strong generalization is considered more robust, harder and more realistic than
    :class:`recpack.splitters.scenarios.WeakGeneralization`

    Args:
        num_users_test: Number of users to assign to the test set.
        num_users_val: Number of users to assign to the validation set.
        num_users_est: Number of users to assign to the estimation set.
        bin_width: Width of the bins to stratify the users into.
        bin_range: Range of the bins to stratify the users into.
        frac_interactions_in: Fraction of interactions to assign to the fold-in set.
        validation: Whether to include a validation set.
        estimation: Whether to include an estimation set.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_users_test: int,
        num_users_val: int,
        num_users_est: int,
        bin_width: int,
        bin_range: Tuple[int, int],
        frac_interactions_in: float = 0.8,
        validation: bool = False,
        estimation: bool = False,
        seed: int = None,
    ):
        if seed is None:
            # Set seed if it was not set before.
            seed = int(np.random.get_state()[1][0])

        self.seed = seed
        self.validation = validation
        self.num_users_test = num_users_test
        self.num_users_val = num_users_val
        self.num_users_est = num_users_est
        self.bin_width = bin_width
        self.bin_range = bin_range
        self.frac_interactions_in = frac_interactions_in
        self.estimation = estimation

        # Set the numpy seed for reproducibility
        np.random.seed(self.seed)

        self.strat_strong_gen_test = ProportionalStratifiedStrongGeneralizationSplitter(
            self.num_users_test, self.bin_width, self.bin_range, seed=self.seed
        )
        self.interaction_split = FractionInteractionSplitter(
            self.frac_interactions_in, seed=self.seed
        )
        if self.estimation:
            if self.num_users_est <= 0:
                raise ValueError(
                    "num_users_est must be greater than 0 when requesting an estimation split."
                )
            self.strat_strong_gen_est = ProportionalStratifiedStrongGeneralizationSplitter(
                self.num_users_est, self.bin_width, self.bin_range, seed=self.seed
            )
        if self.validation:
            assert (
                self.num_users_val > 0
            ), "num_users_val must be greater than 0 when requesting a validation split."
            self.strong_gen_val = StrongGeneralizationSplitter(
                self.num_users_val, seed=self.seed
            )

    def _split(self, data: InteractionMatrix) -> None:
        """Splits your data so that a user can only be in one of
            training, validation or test set.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        """
        if (
            self.num_users_test + self.num_users_val + self.num_users_est
            >= data.num_active_users
        ):
            raise ValueError(
                f"num_users_test ({self.num_users_test}) + num_users_val ({self.num_users_val} + num_users_est ({self.num_users_est})) is greater than the number of users in the data ({data.num_active_users})."
            )

        # split the train and test users from the full dataset
        self._full_train_X, test_data_bins = self.strat_strong_gen_test.split(data)

        self._test_data_bins = [[], []]
        test_data_in_interactions, test_data_out_interactions = [], []
        for test_data_bin in test_data_bins:
            test_data_in_bin, test_data_out_bin = self.interaction_split.split(
                test_data_bin
            )
            self._test_data_bins[0].append(test_data_in_bin)
            self._test_data_bins[1].append(test_data_out_bin)

            for uid, interaction_ids in test_data_in_bin.interaction_history:
                test_data_in_interactions.extend(interaction_ids)
            for uid, interaction_ids in test_data_out_bin.interaction_history:
                test_data_out_interactions.extend(interaction_ids)

        self._test_data_in = data.interactions_in(test_data_in_interactions)
        self._test_data_out = data.interactions_in(test_data_out_interactions)
        self._test_data = (self._test_data_in, self._test_data_out)

        # split the train and validation users from the training dataset
        if self.validation:
            (
                self._validation_train_X,
                validation_data,
            ) = self.strong_gen_val.split(self._full_train_X)

            (
                self._validation_data_in,
                self._validation_data_out,
            ) = self.interaction_split.split(validation_data)

        if self.estimation:
            (
                self._validation_train_X,
                estimation_data_bins,
            ) = self.strat_strong_gen_est.split(self._validation_train_X)

            self._estimation_data_bins = [[], []]
            estimation_data_in_interactions, estimation_data_out_interactions = [], []
            for estimation_data_bin in estimation_data_bins:
                (
                    estimation_data_in_bin,
                    estimation_data_out_bin,
                ) = self.interaction_split.split(estimation_data_bin)
                self._estimation_data_bins[0].append(estimation_data_in_bin)
                self._estimation_data_bins[1].append(estimation_data_out_bin)

                for uid, interaction_ids in estimation_data_in_bin.interaction_history:
                    estimation_data_in_interactions.extend(interaction_ids)
                for uid, interaction_ids in estimation_data_out_bin.interaction_history:
                    estimation_data_out_interactions.extend(interaction_ids)

            self._estimation_data_in = data.interactions_in(
                estimation_data_in_interactions
            )
            self._estimation_data_out = data.interactions_in(
                estimation_data_out_interactions
            )
            self._estimation_data = (
                self._estimation_data_in,
                self._estimation_data_out,
            )

    @property
    def test_data_in(self):
        """Fold-in part of the test dataset"""
        return self.test_data[0]

    @property
    def test_data_out(self):
        """Held-out part of the test dataset"""
        return self.test_data[1]

    @property
    def test_data(self) -> Tuple[InteractionMatrix, InteractionMatrix]:
        return self._test_data

    @property
    def test_data_bins(self) -> Tuple[list, list]:
        return self._test_data_bins

    @property
    def test_data_in_bins(self):
        """Fold-in part of the test dataset"""
        return self.test_data_bins[0]

    @property
    def test_data_out_bins(self):
        """Held-out part of the test dataset"""
        return self.test_data_bins[1]

    @property
    def estimation_data_in(self):
        """Fold-in part of the test dataset"""
        return self.estimation_data[0]

    @property
    def estimation_data_out(self):
        """Held-out part of the test dataset"""
        return self.estimation_data[1]

    @property
    def estimation_data(
        self,
    ) -> Union[Tuple[InteractionMatrix, InteractionMatrix], None]:
        if not self.estimation:
            raise KeyError("This scenario was created without estimation_data.")

        if not hasattr(self, "_estimation_data_in"):
            raise KeyError(
                "Split before trying to access the estimation_data property."
            )
        return self._estimation_data

    @property
    def estimation_data_bins(self) -> Tuple[list, list]:
        return self._estimation_data_bins

    @property
    def estimation_data_in_bins(self):
        """Fold-in part of the estimation dataset"""
        return self._estimation_data_bins[0]

    @property
    def estimation_data_out_bins(self):
        """Held-out part of the estimation dataset"""
        return self._estimation_data_bins[1]

    def _check_split(self):
        """Checks that the splits have been done properly.

        Makes sure all expected attributes are set.
        """
        assert hasattr(self, "_full_train_X") and self._full_train_X is not None
        if self.validation:
            assert (
                hasattr(self, "_validation_train_X")
                and self._validation_train_X is not None
            )
            assert (
                hasattr(self, "_validation_data_in")
                and self._validation_data_in is not None
            )
            assert (
                hasattr(self, "_validation_data_out")
                and self._validation_data_out is not None
            )

        assert hasattr(self, "test_data_bins") and self._test_data_bins is not None
        assert hasattr(self, "_test_data") and self._test_data is not None

        self._check_size()

    def _check_size(self):
        """
        Warns user if any of the sets is unusually small or empty
        """
        n_train = self._full_train_X.num_interactions
        n_test_in = self._test_data_in.num_interactions
        n_test_out = self._test_data_out.num_interactions
        n_test = n_test_in + n_test_out
        n_total = n_train + n_test

        if self.validation:
            n_val_in = self._validation_data_in.num_interactions
            n_val_out = self._validation_data_out.num_interactions
            n_val_train = self._validation_train_X.num_interactions
            n_val = n_val_in + n_val_out
            n_total += n_val

        def check(name, count, total, threshold):
            if (count + 1e-9) / (total + 1e-9) < threshold:
                warn(f"{name} resulting from {type(self).__name__} is unusually small.")

        check("Training set", n_train, n_total, 0.05)
        check("Test set", n_test, n_total, 0.01)
        if self.validation:
            check("Validation set", n_val, n_total, 0.01)
            check("Validation train set", n_val_train, n_train, 0.05)
            check("Validation in set", n_val_in, n_val, 0.05)
            check("Validation out set", n_val_out, n_val, 0.01)
