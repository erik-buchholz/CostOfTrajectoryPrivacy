#!/usr/bin/env python3
"""Adapter to transfer our datasets to the PrivTrace Format and Back.

Refer to `notebooks/privtrace.ipynb` for the usage of this adapter.

"""
from pathlib import Path
from typing import Union

import numpy as np
from pydantic import validate_call

from privtrajgen.datasets.padding import resample_trajectory


@validate_call(config={"arbitrary_types_allowed": True})
def check_valid_lon_lat(data: np.ndarray) -> None:
    """
    Check if the data contains valid longitude and latitude values.

    :param data: numpy.ndarray, shape (n_trajectories, 2, length), containing trajectories with (lon, lat) coordinates
    :raises ValueError: If the data contains invalid longitude or latitude values
    """
    eps = 1e-7
    lon = data[:, 0, :]
    lat = data[:, 1, :]

    if np.any((lon < -180 - eps) | (lon > 180 + eps)):
        raise ValueError(f"Longitude values should be in [-180, 180]. Max abs value found: {np.max(np.abs(lon)):.7f}.")
    if np.any((lat < -90 - eps) | (lat > 90 + eps)):
        raise ValueError(f"Latitude values should be in [-90, 90]. Max abs value found: {np.max(np.abs(lat)):.7f}.")


@validate_call(config={"arbitrary_types_allowed": True})
def transform_data_from_difftraj_to_privtrace(data: np.ndarray, file_path: Union[str, Path]) -> None:
    """
    Transforms trajectory data from DiffTraj format to PrivTrace format and writes it to a file.

    :param data: numpy.ndarray, shape (n_trajectories, 2, length), containing trajectories with (lon, lat) coordinates.
        The data should NOT be normalised, i.e., it should be in the original coordinate system. Longitude first,
        then latitude.
    :param file_path: str or pathlib.Path, path to the output file where the transformed data will be written
    :return: None
    """
    # Validate that data is in the expected format
    if data.ndim != 3 or data.shape[1] != 2:
        raise ValueError(f"Expected data shape (n_trajectories, 2, length), got {data.shape}.")

    # PrivTrace appears to use un-normalised coordinates -> Check that data has been de-normalised
    # I.e., raise an error if data bounded by [-1, 1] or [0, 1]
    eps = 1e-7
    if np.all(np.abs(data) <= 1.0 + eps):
        raise ValueError("Data appears to be normalised. Please denormalise the data before saving.")

    # Assert points are valid lon/lat coordinates
    check_valid_lon_lat(data)

    file_path = Path(file_path)
    with file_path.open('w') as f:
        for idx, traj in enumerate(data):
            f.write(f"#{idx}:\n")
            f.write(">0:")
            # The DL models use float32 --> 7 digits should be enough
            points = [f"{x:.7f},{y:.7f}" for x, y in zip(traj[0], traj[1])]
            f.write(";".join(points) + ";\n")


@validate_call(config={"arbitrary_types_allowed": True})
def transform_data_from_privtrace_to_difftraj(file_path: Union[str, Path], max_len: int = -1) -> np.ndarray:
    """
    Transforms trajectory data from PrivTrace format to DiffTraj format and returns it as a numpy array.

    :param file_path: str or pathlib.Path, path to the input file containing PrivTrace-formatted trajectory data
    :param max_len: int, maximum length of the trajectories. If -1, use the length of the longest trajectory.
    :return: numpy.ndarray, shape (n_trajectories, 2, length), containing trajectories with (lon, lat) coordinates
    """
    file_path = Path(file_path)
    trajectories = []

    with file_path.open('r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if not lines[i].startswith("#"):
            raise ValueError(f"Expected line starting with '#', got: {lines[i]}")
        if i + 1 >= len(lines) or not lines[i + 1].startswith(">0:"):
            raise ValueError(
                f"Expected line starting with '>0:' after line {i}, got: {lines[i + 1] if i + 1 < len(lines) else 'EOF'}")

        points_str = lines[i + 1][3:].strip().rstrip(";")
        points = points_str.split(";")
        x_list, y_list = zip(*(map(float, point.split(",")) for point in points))
        trajectories.append(np.array([x_list, y_list]))

        i += 2

    # This does not work because PrivTrace creates trajectories of different lengths --> Resample using
    # resample_trajectory(x: np.ndarray, length: int)
    if max_len == -1:
        max_len = max(traj.shape[1] for traj in trajectories)

    # Resample trajectories to the same length using 'resample_trajectory'
    trajectories = [resample_trajectory(traj.T, length=max_len).T for traj in trajectories]

    dataset = np.stack(trajectories, axis=0)

    # Check that the data is in the correct range for lon/lat
    check_valid_lon_lat(dataset)

    # Check expected shape
    assert dataset.ndim == 3 and dataset.shape[1] == 2, \
        f"Expected data shape (n_trajectories, 2, length), got {dataset.shape}."

    return dataset
