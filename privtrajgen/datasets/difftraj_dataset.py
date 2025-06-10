#!/usr/bin/env python3
"""Wrapper for the TrajectoryDataset providing conditional attributes."""
import logging
import os
from typing import Union

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm.autonotebook import tqdm

from privtrajgen import config
from privtrajgen.config import Datasets
from .base_dataset import TrajectoryDataset
from .constants import number_of_trajectories, trajectory_lengths
from .padding import resample_trajectory

log = logging.getLogger(__name__)

# Data Directory
DT_DATA_DIR = os.path.join(config.BASE_DIR, "data", "processed")
# Path to the npy files containing the preprocessed DiffTraj datasets
difftraj_paths: dict[Union[Datasets, str], str] = {
    Datasets.FSNYC: os.path.join(DT_DATA_DIR, "fs_normalised_interpolated.npy"),
    Datasets.GEOLIFE: os.path.join(DT_DATA_DIR, "geolife_normalised_interpolated.npy"),
    Datasets.PORTO: os.path.join(DT_DATA_DIR, "porto_normalised_interpolated.npy"),
}
# Path to the npy files containing the conditional attributes in the same order
difftraj_conditional_paths = {
    Datasets.FSNYC: os.path.join(DT_DATA_DIR, "fs_head.npy"),
    Datasets.GEOLIFE: os.path.join(DT_DATA_DIR, "geolife_head.npy"),
    Datasets.PORTO: os.path.join(DT_DATA_DIR, "porto_head.npy"),
}


def dataset_to_difftraj_npy(dataset: TrajectoryDataset, path: str, cond_path: str) -> None:
    """Convert a TrajectoryDataset to a numpy array in the DiffTraj format and save it to disk."""
    assert (dataset.return_conditional == True) or cond_path is None, "Dataset must return conditional information."

    # Sort the dataset by TIDs to make sure the order is reproducible as the npy does not have any IDs
    sorted_tids = sorted(dataset.tids)
    max_length = dataset.max_len

    ############ Trajectory Dataset ############
    # Interpolate all trajectories to the same length (TIDs are sorted, so we can just iterate over the dataset)
    interpolated_trajs = np.array([resample_trajectory(dataset.get_encoding(tid), length=max_length) for tid in
                                   tqdm(sorted_tids, total=len(sorted_tids), desc="Interpolating trajectories")])
    # Those are already normalised
    normalised_trajs = interpolated_trajs
    assert normalised_trajs.min() >= -1 + (-1e-8) and normalised_trajs.max() <= 1, (
        f"Expected [-1, 1], got [{normalised_trajs.min()}, {normalised_trajs.max()}]."
    )
    # For DiffTraj, we need to transpose, as the expected format is (B, 2, 100)
    normalised_trajs = normalised_trajs.transpose(0, 2, 1)
    assert normalised_trajs.shape == (len(dataset), 2, max_length)
    if path != difftraj_paths[dataset.name]:
        log.warning(f"Path {path} does not match the default path for dataset ({difftraj_paths[dataset.name]}).")
    np.save(path, normalised_trajs)
    log.info(f"Saved DiffTraj dataset to {path}.")

    ############ Conditional Information Dataset ############
    if cond_path is not None:
        conditional_info = np.array([dataset.get_conditional_info(tid=tid) for tid in
                                     tqdm(sorted_tids, total=len(sorted_tids),
                                          desc="Collecting Conditional Information")])
        # Assert that attributes 1:6 are normalised, i.e., the mean is approx 0 and the std is 1
        assert np.allclose(conditional_info[:, 1:6].mean(axis=0), 0,
                           atol=1e-2), f"Mean of normalised attributes is not close to 0: {conditional_info[:, 1:6].mean(axis=0)}."
        assert np.allclose(conditional_info[:, 1:6].std(axis=0), 1,
                           atol=1e-2), f"Standard deviation of normalised attributes is not close to 1: {conditional_info[:, 1:6].std(axis=0)}."
        if cond_path != difftraj_conditional_paths[dataset.name]:
            log.warning(
                f"Path {cond_path} does not match the default conditional path for dataset ({difftraj_conditional_paths[dataset.name]}).")
        np.save(cond_path, conditional_info)
        log.info(f"Saved head dataset to {cond_path}.")


def get_difftraj_dataset(dataset_name: Datasets, conditional: bool, as_numpy: bool = False) -> TensorDataset:
    """
    Return a Dataset in the format used by the DiffTraj model.
    dataset.shape = (n_samples, 2, n_timesteps)
    :param dataset_name: The name of the dataset to load
    :param conditional: Whether the dataset is conditional or not
    :param as_numpy: Whether to return the dataset as a numpy array or a TensorDataset
    :return: A TensorDataset containing the data and the conditional information
    """
    filepath = difftraj_paths[dataset_name]
    data = torch.from_numpy(np.load(filepath))

    # Assert that the dataset has the expected shape
    assert data.shape[0] == number_of_trajectories[dataset_name], (
        f"Expected {number_of_trajectories[dataset_name]} trajectories, got {data.shape[0]}.")
    assert data.shape[1] == 2, f"Expected 2 spatial dimensions, got {data.shape[1]}."
    assert data.shape[2] == trajectory_lengths[dataset_name], (
        f"Expected {trajectory_lengths[dataset_name]} timesteps, got {data.shape[2]}.")

    if conditional:
        head = np.load(difftraj_conditional_paths[dataset_name])
        # Make sure that the head is normalised (only columns 1:6)
        assert np.allclose(head[:, 1:6].mean(axis=0), 0, atol=1e-2), (
            f"Mean of conditional attributes is not close to 0: {head[:, 1:6].mean(axis=0)}.")
        assert np.allclose(head[:, 1:6].std(axis=0), 1, atol=1e-2), (
            f"Standard deviation of conditional attributes is not close to 1: {head[:, 1:6].std(axis=0)}.")
        head = torch.from_numpy(head)
    else:
        head = torch.zeros(data.shape[0], 1)
    assert data.shape[0] == head.shape[0], (
        f"Data and head have different number of samples: {data.shape[0]} != {head.shape[0]}.")
    if as_numpy:
        return data.numpy(), head.numpy()
    else:
        dataset = TensorDataset(data, head)
    return dataset


def get_conditional_dataset(dataset: Datasets) -> torch.Tensor:
    """Return the conditional information only for validation"""
    path = difftraj_conditional_paths[dataset]
    head = np.load(path)

    # Assert Shape
    assert head.shape[0] == number_of_trajectories[dataset], (
        f"Expected {number_of_trajectories[dataset]} trajectories, got {head.shape[0]}.")
    assert head.shape[1] == 8, f"Expected 8 attributes, got {head.shape[1]}."

    return torch.from_numpy(head)
