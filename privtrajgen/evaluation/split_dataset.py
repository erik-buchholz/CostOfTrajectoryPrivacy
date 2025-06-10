#!/usr/bin/env python3
"""Pre-computes indices for the train and test sets using X-fold cross-validation"""
import argparse
import logging
import os
from typing import Tuple, List, Any, Union, Sequence, cast

import numpy as np
from pydantic import validate_call
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import TensorDataset

from privtrajgen.config import DATA_DIR, Datasets
from privtrajgen.datasets import number_of_trajectories, TrajectoryDataset
from privtrajgen.datasets.split_dataset import SubTrajectoryDataset

log = logging.getLogger(__name__)


def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


@validate_call(config={"arbitrary_types_allowed": True})
def get_train_set_len(dataset: Datasets, n_splits: int = 5) -> int:
    """Get the length of the training set for a specific split without loading the indices!

    :param dataset: The dataset to split
    :param n_splits: The number of splits to create

    :return: The length of the training set
    """
    # Get the dataset's size
    num_traj = number_of_trajectories[dataset]

    # Handle the case where no split is used
    if n_splits <= 1:
        return num_traj

    # Compute the length of the training set
    train_len = num_traj * (n_splits - 1) // n_splits

    return train_len


@validate_call(config={"arbitrary_types_allowed": True})
def get_test_set_len(dataset: Datasets, n_splits: int = 5) -> int:
    """Get the length of the test set for a specific split without loading the indices!

    :param dataset: The dataset to split
    :param n_splits: The number of splits to create

    :return: The length of the test set
    """
    # Get the dataset's size
    num_traj = number_of_trajectories[dataset]

    # Handle the case where no split is used
    if n_splits <= 1:
        return 0

    # Compute the length of the test set
    test_len = num_traj // n_splits

    return test_len


@validate_call(config={"arbitrary_types_allowed": True})
def split_dataset(dataset: Datasets, n_splits: int = 5, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pre-computes indices for the train and test sets using X-fold cross-validation

    Path Structure: DATA_DIR/folds/{dataset}/{n_splits}-fold/(train|test)_{split}.npy

    :param dataset: The dataset to split
    :param n_splits: The number of splits to create
    :param random_state: The random seed to use

    :return: A list of tuples containing the indices for the train and test sets
    """
    # Get the dataset's size
    num_traj = number_of_trajectories[dataset]

    # Create the KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Create directory structure
    fold_dir = os.path.join(DATA_DIR, 'folds', dataset, f'{n_splits}-fold')
    os.makedirs(fold_dir, exist_ok=True)

    # Create the splits
    splits = []
    for i, (train_index, test_index) in enumerate(kf.split(np.arange(num_traj))):
        splits.append((train_index, test_index))

        # Create the paths
        train_path = os.path.join(fold_dir, f'train_{i}.npy')
        test_path = os.path.join(fold_dir, f'test_{i}.npy')

        # Save the splits
        np.save(train_path, train_index)
        np.save(test_path, test_index)
        logc(f"Saved split {i} for {dataset} to disk.")

    return splits


@validate_call(config={"arbitrary_types_allowed": True})
def get_train_set_idx(dataset: Datasets, split: int, n_splits: int = 5) -> np.ndarray:
    """Get the training set for a specific split

    :param dataset: The dataset to split
    :param split: The split to get the training set for \\in [0, n_splits)
    :param n_splits: The number of splits to create

    :return: The training set indices
    """
    # Load the training set
    train_path = os.path.join(DATA_DIR, 'folds', dataset, f'{n_splits}-fold', f'train_{split}.npy')
    train_set = np.load(train_path)
    return train_set


@validate_call(config={"arbitrary_types_allowed": True})
def get_test_set_idx(dataset: Datasets, split: int, n_splits: int = 5) -> np.ndarray:
    """Get the test set for a specific split

    :param dataset: The dataset to split
    :param split: The split to get the test set for \\in [0, n_splits)
    :param n_splits: The number of splits to create

    :return: The test set indices
    """
    # Load the test set
    test_path = os.path.join(DATA_DIR, 'folds', dataset, f'{n_splits}-fold', f'test_{split}.npy')
    test_set = np.load(test_path)
    return test_set


@validate_call(config={"arbitrary_types_allowed": True})
def get_train_test_set_idx(dataset: Datasets, split: int, n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Get the training and test sets for a specific split

    :param dataset: The dataset to split
    :param split: The split to get the training and test sets for
    :param n_splits: The number of splits to create

    :return: A tuple containing the training and test set indices
    """
    train_set = get_train_set_idx(dataset, split, n_splits)
    test_set = get_test_set_idx(dataset, split, n_splits)
    return train_set, test_set


dataset_types = Union[Tensor, TensorDataset, np.ndarray, TrajectoryDataset]


@validate_call(config={"arbitrary_types_allowed": True})
def select_indices(indices: Union[Sequence[int], np.ndarray], dataset: dataset_types) -> dataset_types:
    """Select a subset of indices from a dataset

    :param indices: The indices to select
    :param dataset: The dataset to select from

    :return: The selected dataset
    """
    if isinstance(dataset, (Tensor, np.ndarray)):
        return dataset[indices]  # type: ignore
    elif isinstance(dataset, TensorDataset):
        return TensorDataset(*[t[indices] for t in dataset.tensors])
    elif isinstance(dataset, TrajectoryDataset):
        if isinstance(indices, np.ndarray):
            indices = cast(Sequence[int], indices.tolist())
        return SubTrajectoryDataset(dataset, indices)
    else:
        raise ValueError(f"Unknown dataset type {type(dataset)}")


@validate_call(config={"arbitrary_types_allowed": True})
def get_set(dataset_name: Datasets, split: int, data: dataset_types, train: bool, n_splits: int = 5) -> dataset_types:
    """Get the training or test set for a specific split

    :param dataset_name: The dataset to split
    :param split: The split to get the training or test set for
    :param data: The data to split
    :param train: Whether to get the training set
    :param n_splits: The number of splits to create. Special case: if n_splits == 1, the full dataset is used.

    :return: The training or test set
    """
    if n_splits == 1:
        log.warning("Not using cross-validation.")
        return data
    if train:
        indices = get_train_set_idx(dataset_name, split, n_splits)
    else:
        indices = get_test_set_idx(dataset_name, split, n_splits)
    return select_indices(indices, data)



@validate_call(config={"arbitrary_types_allowed": True})
def get_train_set(dataset_name: Datasets, split: int, data: dataset_types, n_splits: int = 5) -> dataset_types:
    """Get the training set for a specific split

    :param dataset_name: The dataset to split
    :param split: The split to get the training set for
    :param data: The data to split
    :param n_splits: The number of splits to create. Special case: if n_splits == 1, the full dataset is used.

    :return: The training set
    """
    return get_set(dataset_name, split, data, train=True, n_splits=n_splits)


@validate_call(config={"arbitrary_types_allowed": True})
def get_test_set(dataset_name: Datasets, split: int, data: dataset_types, n_splits: int = 5) -> dataset_types:
    """Get the test set for a specific split

    :param dataset_name: The dataset to split
    :param split: The split to get the test set for
    :param data: The data to split
    :param n_splits: The number of splits to create. Special case: if n_splits == 1, the full dataset is used.

    :return: The test set
    """
    return get_set(dataset_name, split, data, train=False, n_splits=n_splits)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Pre-computes indices for the train and test sets using X-fold cross-validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", type=Datasets, choices=list(Datasets), help="The dataset to split")
    parser.add_argument("-k", "--n_splits", type=int, default=5, help="The number of splits to create")
    parser.add_argument("-r", "--random_state", type=int, default=42, help="The random seed to use")

    # Parse arguments
    args = parser.parse_args()

    # Split the dataset
    splits = split_dataset(args.dataset, args.n_splits, args.random_state)
