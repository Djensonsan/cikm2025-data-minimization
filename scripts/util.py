import os
import yaml
import datetime
from typing import List, Union, Tuple
from scipy.sparse import csr_matrix
from recpack.datasets import Netflix, MovieLens25M, MillionSongDataset, DummyDataset
from recpack.preprocessing.filters import MinUsersPerItem, MinItemsPerUser, MinRating, Deduplicate

def load_config(file_path):
    """Load a YAML file from the given file path."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_dataset(dataset_path, dataset):
    # Standard preprocessing for MSD:
    MSD = MillionSongDataset(path=dataset_path, use_default_filters=False)
    MSD.add_filter(MinUsersPerItem(200, MSD.ITEM_IX, MSD.USER_IX))
    MSD.add_filter(MinItemsPerUser(20, MSD.ITEM_IX, MSD.USER_IX))
    MSD.add_filter(Deduplicate(MSD.USER_IX, MSD.ITEM_IX))

    # Standard preprocessing for MovieLens:
    ML25 = MovieLens25M(path=dataset_path, use_default_filters=False)
    ML25.add_filter(MinRating(4, ML25.RATING_IX))
    ML25.add_filter(MinUsersPerItem(5, ML25.ITEM_IX, ML25.USER_IX))
    ML25.add_filter(MinItemsPerUser(5, ML25.ITEM_IX, ML25.USER_IX))
    ML25.add_filter(Deduplicate(ML25.USER_IX, ML25.ITEM_IX))

    datasets = {
        "Netflix": {"dataset": Netflix(path=dataset_path)},
        "MovieLens": {"dataset": ML25},
        "MSD": {"dataset": MSD},
        "Dummy": {
            "dataset": DummyDataset(
                num_users=100, num_items=100, num_interactions=10000
            )
        },
    }
    return datasets[dataset]

def eliminate_empty_users(data_in: Union[List[csr_matrix], csr_matrix], data_out: Union[List[csr_matrix], csr_matrix]) -> Tuple[Union[List[csr_matrix], csr_matrix], Union[List[csr_matrix], csr_matrix]]:
    """Eliminate users that have no interactions in ``data_out``.

    We cannot make accurate predictions of interactions for
    these users as there are none.

    :param data_out: ground-truth interactions.
    :type data_out: Union[List[csr_matrix], csr_matrix]
    :param data_in: input interactions.
    :type data_in: Union[List[csr_matrix], csr_matrix]
    :return: (y_true, y_pred), with zero users eliminated.
    :rtype: Union[List[csr_matrix], csr_matrix]]
    """
    if isinstance(data_out, csr_matrix):
        nonzero_users = list(set(data_out.nonzero()[0]))
        return data_in[nonzero_users, :], data_out[nonzero_users, :]
    elif isinstance(data_out, list):
        data_in_list = []
        data_out_list = []
        for i, (data_in_i, data_out_i) in enumerate(zip(data_in, data_out)):
            nonzero_users = list(set(data_out_i.nonzero()[0]))
            data_in_list.append(data_in_i[nonzero_users, :])
            data_out_list.append(data_out_i[nonzero_users, :])
        return data_in_list, data_out_list

def get_script_name(script_file):
    """Extract the base name of the script from the passed file path, without its directory or file extension."""
    return os.path.splitext(os.path.basename(script_file))[0]

def get_config_path():
    """Get the default path to the configuration files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config')
    return config_path

def ensure_directory_exists(path):
    """Ensure that the directory exists, and if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


def construct_results_directory_path(
    output_path, script_name, dataset, algorithm, scenario
):
    """Construct the path for output files based on the input parameters and the name of the script."""
    # Pattern: YYYY-MM-DD_HH-MM
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_dir = os.path.join(
        output_path, script_name, dataset, scenario, algorithm, date_str
    )

    # Ensure the full path for results exists
    ensure_directory_exists(results_dir)

    return results_dir
