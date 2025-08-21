import logging
import random
import pandas as pd
from typing import Tuple, Union, Dict, List, Optional, Any
import numpy as np
from recpack.scenarios.splitters import Splitter, UserSplitter
from recpack.matrix import InteractionMatrix

logger = logging.getLogger('minipack')


class StrongGeneralizationSplitter(Splitter):
    def __init__(self, num_out_users: int = 1000, seed: int = None):
        self.num_out_users = num_out_users
        if seed is None:
            # Set seed if it was not set before.
            seed = int(np.random.get_state()[1][0])
        self.seed = seed

    def split(self, data) -> Tuple[InteractionMatrix, InteractionMatrix]:
        # Validate that there are enough users to perform the split
        if self.num_out_users >= data.num_active_users:
            raise ValueError(
                f"num_out_users ({self.num_out_users}) is greater than the number of users in the data ({data.num_active_users})."
            )

        users = set(data._df[data.USER_IX])

        # Sample users
        random.seed(self.seed)
        users_out = set(random.sample(list(users), self.num_out_users))

        users_in = users - users_out

        # Perform user-based data split
        u_splitter = UserSplitter(users_in, users_out)
        data_in, data_out = u_splitter.split(data)

        logger.info(f"{self.identifier} - Split successful")

        return data_in, data_out

class ProportionalStratifiedStrongGeneralizationSplitter(Splitter):
    def __init__(
        self,
        num_out_users=1000,
        bin_width=None,
        bin_range=None,
        seed=None,
    ):
        if bin_width is None or bin_range is None:
            bin_width = None
            bin_range = None
        if (bin_width is not None) and (bin_range is not None):
            if not bin_width > 0:
                raise ValueError("bin_width must be greater than 0")
            if not bin_range[0] < bin_range[1]:
                raise ValueError(
                    "bin_range must be a tuple with the first element smaller than the second"
                )
            if not bin_range[0] >= 0:
                raise ValueError(
                    "bin_range must be a tuple with both elements greater than or equal to 0"
                )
            if not bin_width <= bin_range[1]:
                raise ValueError(
                    "bin_width must be smaller or equal to the second element of bin_range"
                )
            if not (bin_range[1] - bin_range[0]) % bin_width == 0:
                raise ValueError("bin_range must be divisible by bin_width")

        self.num_out_users = num_out_users
        self.num_bins = (bin_range[1] - bin_range[0]) / bin_width
        self.bin_width = bin_width
        self.bin_range = bin_range
        self.bins = np.arange(bin_range[0], bin_range[1] + bin_width, bin_width)

        # TODO: Dead code. Numpy seed is set in the Scenario class, not used in this class.
        if seed is None:
            # Set seed if it was not set before.
            seed = int(np.random.get_state()[1][0])
        self.seed = seed


    def split(self, data) -> Tuple[InteractionMatrix, List[InteractionMatrix]]:
        # check that there's enough users to split
        if self.num_out_users >= data.num_active_users:
            raise ValueError(
                f"num_out_users ({self.num_out_users}) is greater than the number of users in the data ({data.num_active_users})."
            )
        # Count interactions per user
        interactions_per_user = data._df.groupby(data.USER_IX)[data.ITEM_IX].count()

        # Calculate the number of users in each bin using pd.cut
        users_per_bin = pd.cut(interactions_per_user, bins=self.bins, right=False, labels=False)
        users_per_bin = users_per_bin.value_counts().reindex(range(len(self.bins) - 1), fill_value=0)

        # Creating bindex
        users_per_bin.index = range(len(users_per_bin))

        # Calculate the overall fractions and ensure they sum to num_out_users
        user_fractions_per_bin = users_per_bin / users_per_bin.sum()
        users_to_sample_per_bin = np.floor(self.num_out_users * user_fractions_per_bin).astype(int)

        # Adjust the last element to correct the sum to num_out_users
        users_to_sample_per_bin.iloc[-1] += self.num_out_users - users_to_sample_per_bin.sum()

        # Initialize variables for binning process
        remaining_users = set(interactions_per_user.index)
        data_in = data
        data_out_partitions = []  # Collects out data partitions

        # Iteratively create data partitions
        for i in range(len(self.bins) - 1):
            lower_bound = self.bins[i]
            upper_bound = self.bins[i + 1]

            # Identify users within the current bin's interaction count range
            users_in_bin = interactions_per_user[
                (interactions_per_user >= lower_bound) & (interactions_per_user < upper_bound)
                ].index

            # Ensure there are enough users in the bin
            if len(users_in_bin) < users_to_sample_per_bin[i]:
                raise ValueError(
                    f"Bin {i} does not have enough users to sample from."
                )

            # Sample users from the bin
            sampled_users = np.random.choice(users_in_bin, size=users_to_sample_per_bin[i], replace=False)
            remaining_users -= set(sampled_users)

            # Perform user-based data split
            u_splitter = UserSplitter(remaining_users, sampled_users)
            data_in, data_out_partition = u_splitter.split(data_in)
            data_out_partitions.append(data_out_partition)

        logger.info(f"{self.identifier} - Split successful")

        return data_in, data_out_partitions

class UniformlyStratifiedStrongGeneralizationSplitter(Splitter):
    def __init__(
        self,
        num_out_users=1000,
        bin_width=None,
        bin_range=None,
        seed=None,
    ):
        if bin_width is None or bin_range is None:
            bin_width = None
            bin_range = None
        if (bin_width is not None) and (bin_range is not None):
            if not bin_width > 0:
                raise ValueError("bin_width must be greater than 0")
            if not bin_range[0] < bin_range[1]:
                raise ValueError(
                    "bin_range must be a tuple with the first element smaller than the second"
                )
            if not bin_range[0] >= 0:
                raise ValueError(
                    "bin_range must be a tuple with both elements greater than or equal to 0"
                )
            if not bin_width <= bin_range[1]:
                raise ValueError(
                    "bin_width must be smaller or equal to the second element of bin_range"
                )
            if not (bin_range[1] - bin_range[0]) % bin_width == 0:
                raise ValueError("bin_range must be divisible by bin_width")

        self.num_out_users = num_out_users
        self.num_bins = (bin_range[1] - bin_range[0]) / bin_width
        self.users_per_bin = int(num_out_users / self.num_bins)
        self.bin_width = bin_width
        self.bin_range = bin_range

        if seed is None:
            # Set seed if it was not set before.
            seed = int(np.random.get_state()[1][0])
        self.seed = seed


    def split(self, data) -> Tuple[InteractionMatrix, List[InteractionMatrix]]:
        # check that there's enough users to split
        if self.num_out_users >= data.num_active_users:
            raise ValueError(
                f"num_out_users ({self.num_out_users}) is greater than the number of users in the data ({data.num_active_users})."
            )
        # Aliases for readability
        user_id_column = data.USER_IX
        item_id_column = data.ITEM_IX

        # Count interactions per user
        interactions_per_user = data._df.groupby(user_id_column)[item_id_column].count()

        # Initialize variables for binning process
        bindex = 0
        remaining_users = set(interactions_per_user.index)
        data_in = data
        data_out_partitions = []  # Collects out data partitions

        # Iteratively create data partitions until bin range is exhausted
        while self.bin_range[0] + ((bindex + 1) * self.bin_width) <= self.bin_range[1]:
            lower_count = self.bin_range[0] + (bindex * self.bin_width)
            upper_count = self.bin_range[0] + ((bindex + 1) * self.bin_width)

            # Identify users within the current bin's interaction count range
            users_in_bin = interactions_per_user[
                (interactions_per_user >= lower_count)
                & (interactions_per_user < upper_count)
            ]

            # Ensure there are enough users in the bin
            if len(users_in_bin) < self.users_per_bin:
                raise ValueError(
                    f"Bin {bindex} does not have enough users to sample from."
                )

            # Sample users from the bin
            sampled_users = users_in_bin.sample(
                n=self.users_per_bin, random_state=self.seed
            ).index
            remaining_users -= set(sampled_users)  # Update remaining users

            # Perform user-based data split
            u_splitter = UserSplitter(remaining_users, sampled_users)
            data_in, data_out_partition = u_splitter.split(data_in)
            data_out_partitions.append(data_out_partition)

            bindex += 1

        logger.info(f"{self.identifier} - Split successful")

        return data_in, data_out_partitions
