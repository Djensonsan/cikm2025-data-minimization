import logging
import numpy as np
from recpack.matrix import InteractionMatrix
from recpack.scenarios import Scenario
from recpack.scenarios.splitters import FractionInteractionSplitter
from minipack.scenarios.splitters import StrongGeneralizationSplitter

logger = logging.getLogger("minipack")


class StrongGeneralization(Scenario):
    """
    Predict (randomly) held-out interactions of previously unseen users.

    During splitting each user is randomly assigned to one of three groups of users: training, validation and testing.

    Test, estimation and validation users' interactions are split into a `data_in` (fold-in) and
    `data_out` (held-out) set.

    ``frac_interactions_in`` of interactions are assigned to fold-in, the remainder
    to the held-out set.

    Strong generalization is considered more robust, harder and more realistic than
    :class:`recpack.splitters.scenarios.WeakGeneralization`

    Args:
        num_users_test: Number of users to assign to the test set.
        num_users_val: Number of users to assign to the validation set.
        frac_interactions_in: Fraction of interactions to assign to the fold-in set.
        validation: Whether to include a validation set.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_users_test: int,
        num_users_val: int,
        frac_interactions_in: float = 0.8,
        validation: bool = False,
        seed: int = None,
    ):
        if seed is None:
            # Set seed if it was not set before.
            seed = int(np.random.get_state()[1][0])
        self.seed = seed
        self.validation = validation
        self.num_users_test = num_users_test
        self.num_users_val = num_users_val
        self.frac_interactions_in = frac_interactions_in

        self.strong_gen_test = StrongGeneralizationSplitter(
            self.num_users_test, seed=self.seed
        )
        self.strong_gen_val = StrongGeneralizationSplitter(
            self.num_users_val, seed=self.seed
        )
        self.interaction_split = FractionInteractionSplitter(
            self.frac_interactions_in, seed=self.seed
        )
        if self.validation:
            assert (
                self.num_users_val > 0
            ), "num_users_val must be greater than 0 when requesting a validation split."
            self.strong_gen_val = StrongGeneralizationSplitter(
                self.num_users_val, seed=self.seed
            )

    def _split(self, data: InteractionMatrix) -> None:
        """
        Splits your data so that a user can only be in one of training, validation or test set.

        :param data: Interaction matrix to be split.
        :type data: InteractionMatrix
        """
        if self.num_users_test + self.num_users_val >= data.num_active_users:
            raise ValueError(
                f"num_users_test ({self.num_users_test}) + num_users_val ({self.num_users_val}) is greater than the number of users in the data ({data.num_active_users})."
            )

        # split the train and test users from the full dataset
        self._full_train_X, test_data = self.strong_gen_test.split(data)

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

        self._test_data_in, self._test_data_out = self.interaction_split.split(
            test_data
        )
