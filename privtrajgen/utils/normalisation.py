#!/usr/bin/env python3
"""Utility functions relating to data processing."""
import logging
from typing import Optional, Union, Dict, Tuple, Sequence, cast, TypeVar

import numpy as np
import pandas as pd
from pydantic import validate_call

from privtrajgen.config import SPATIAL_COLUMNS

log = logging.getLogger(__name__)

# Dataset Type
T_D = TypeVar("T_D", pd.DataFrame, np.ndarray, Sequence[np.ndarray])
# Series/Array Type
S = TypeVar("S", pd.Series, np.ndarray)
T = TypeVar("T", pd.DataFrame, np.ndarray)
C = Union[Sequence[str], Sequence[int]]  # Columns
# Types for reference points and scale factors
R = Union[Dict[Union[int, str], float], Sequence[float]]
OptDR = Dict[Union[int, str], Optional[float]]
DR = Dict[Union[int, str], float]


def make_dict(d: Optional[R], keys: C) -> OptDR:
    """
    Creates a dictionary mapping keys to values from the input reference `d`,
    or assigns None if the key is not valid. Raises an error for invalid keys
    unless `d` is None.

    :param d: A dictionary or sequence containing the reference data. If None,
              all keys will map to None.
    :param keys: A sequence of keys to extract from `d`. Keys must be valid
                 indices if `d` is a sequence, or valid keys if `d` is a dictionary.
    :return: A dictionary where each key in `keys` maps to a value from `d` or None.
    :raises KeyError: If a key is not valid for the given dictionary `d`.
    :raises IndexError: If a key is not a valid index for the given sequence `d`.
    :raises TypeError: If `d` is neither a dictionary nor a sequence.
    """
    result: OptDR

    if d is None:
        result = {key: None for key in keys}
    elif isinstance(d, Sequence):
        result = {keys[i]: d[i] for i in range(len(keys))}
    elif isinstance(d, dict):
        result = cast(OptDR, d)
    else:
        raise TypeError("Unsupported type for 'd'. Must be a dictionary or sequence.")

    return result


@validate_call(config={"arbitrary_types_allowed": True})
def normalize_sequence(series: S,
                       ref: Optional[float] = None,
                       scale_factor: Optional[float] = None,
                       inplace: bool = False) -> Tuple[S, float, float]:
    """
    Normalizes a sequence of values to [-1;1]
    :param series: Series to normalize
    :param ref: (optional) Reference point (default: (max(series) + min(series)) / 2)
    :param scale_factor: (optional) Scale factor (default: max(abs(series - ref)))
    :param inplace: Transform inplace
    :return: Normalized series, reference point, scale factor
    """
    # Check series type
    if type(series) is not np.ndarray and type(series) is not pd.Series:
        raise ValueError(f"Unexpected type '{type(series)}'.")
    # Convert type and copy if not inplace
    series = series.astype(float, copy=(not inplace))
    # Determine reference and scale factor if not provided
    if ref is None:
        ref = (max(series) + min(series)) / 2.0
    # Normalize series
    series -= ref
    # Determine scale factor if not provided
    if scale_factor is None:
        scale_factor = max(abs(series))
    series /= scale_factor
    return series, ref, scale_factor


@validate_call(config={"arbitrary_types_allowed": True})
def denormalize_sequence(series: S,
                         ref: float,
                         scale_factor: float,
                         inplace: bool = False) -> S:
    """
    Denormalizes a sequence of values (inverse operation to normalize_sequence)

    :param series: Series to denormalize
    :param ref: float - Reference Point
    :param scale_factor: float - Scaling Factor
    :param inplace: Transform inplace
    :return: Denormalized series
    """
    # Check series type
    if type(series) is not np.ndarray and type(series) is not pd.Series:
        raise ValueError(f"Unexpected type '{type(series)}'.")

    # Check that inputs are between -1 and 1
    eps = 1e-7
    if np.any(series > 1 + eps) or np.any(series < -1 - eps):
        # We cannot raise an error here because generated data might very well be outside [-1, 1]
        # Yet, we should log a warning to notice issues early on.
        log.warning("Denormalizing a sequence that is not in the range [-1, 1].")
        log.warning(f"Max: {np.max(series)}, Min: {np.min(series)}")

    # Convert type and copy if not inplace
    series = series.astype(float, copy=(not inplace))

    # Denormalize series
    series *= scale_factor
    series += ref

    return series


@validate_call(config={"arbitrary_types_allowed": True})
def normalize_points(df: T,
                     columns: C = ['x', 'y'],
                     ref: Optional[R] = None,
                     scale_factor: Optional[R] = None,
                     inplace: bool = False) -> Tuple[
    T, Dict[Union[int, str], float], Dict[Union[int, str], float]]:
    """
    Scales dataframe or array columns to [-1, 1].

    :param df: DataFrame or ndarray with columns to normalize.
    :param columns: Sequence of columns (DataFrame) or indices (ndarray) to normalize.
    :param ref: Reference points as Sequence/dict, defaults to column mean.
    :param scale_factor: Scale factors as Sequence/dict, defaults to max(abs(series)).
    :param inplace: If True, modify df in place.
    :return: Normalized df, Sequence of reference points, Sequence of scale factors.
    """
    return normalise(df, ref, scale_factor, columns, inplace)


@validate_call(config={"arbitrary_types_allowed": True})
def denormalize_points(df: T,
                       ref: Union[R],
                       scale: Union[R],
                       columns: C = (0, 1),
                       inplace: bool = False) -> T:
    """
    Denormalizes a dataframe (inverse operation to normalize_points)
    :param df: Sequence of points to denormalize
    :param ref: reference point either as Sequence or dict with columns as keys
    :param scale: scale factor either as Sequence or dict with columns as keys
    :param columns: Sequence of columns or array indices to denormalize
    :param inplace: Transform inplace
    :return: denormalized df
    """
    return denormalise(df, ref, scale, columns, inplace)


@validate_call(config={"arbitrary_types_allowed": True})
def normalise(
        dataset: T_D,
        ref: Optional[R] = None,
        scale: Optional[R] = None,
        columns: C = SPATIAL_COLUMNS,
        inplace: bool = False
) -> Tuple[T_D, DR, DR]:
    """
    Normalises a dataset of trajectories or points to [-1, 1] using min-max scaling.

    :param dataset: Dataset to normalise. Shape can be:
        - (n_trajectories, n_points, n_features) for trajectories
        - (n_points, n_features) for points
        - Sequence[np.ndarray], each element with shape (n_points, n_features)
        - pd.DataFrame with columns column names as provided in 'columns'
    :param ref: Reference points as Sequence/dict. If dict, keys are column names. If Sequence, in order of 'columns'.
        If None, defaults to ((max + min) / 2) for each column.
        Example: {'lat': 0.0, 'lon': 0.0}
    :param scale: Scale factors as Sequence/dict. If dict, keys are column names. If Sequence, in order of 'columns'.
        If None, defaults to max(abs(series - ref)) for each column.
        Example: {'lat': 1.0, 'lon': 1.0}
    :param columns: Sequence of columns to normalise. If 'dataset' is a DataFrame, use column names. If ndarray, use
        keys of 'ref' and 'scale' in order of columns.
        E.g. dataset = [ [ [lat, lon], [lat, lon], ...], [ [lat, lon], [lat, lon], ...] ], then columns=['lat', 'lon']
    :param inplace: If True, modify dataset in place.
    :return: Normalised dataset, reference point as dict, scale factor as dict.
    """
    # Assert that columns math the keys of ref and scale
    if (isinstance(ref, dict) and set(ref.keys()) != set(columns)) or (
            isinstance(ref, Sequence) and not isinstance(columns[0], int)) or \
            (isinstance(scale, dict) and set(scale.keys()) != set(columns)) or (
            isinstance(scale, Sequence) and not isinstance(columns[0], int)):
        raise ValueError("Columns must match keys of ref and scale.")
    # Initialise reference and scale as dictionaries if provided as Sequences
    reference_point = make_dict(ref, columns)
    scale_factor = make_dict(scale, columns)
    del ref, scale

    # Make float it not already
    if isinstance(dataset, Sequence):
        dataset = [arr.astype(float, copy=(not inplace)) for arr in dataset]
    else:
        dataset = dataset.astype(float, copy=(not inplace))

    # Define helper to normalise sequences (inplace = True because we already made a copy)
    def normalise_single_column(series: S, col: Union[int, str]) -> Union[
        np.ndarray, pd.Series]:
        return normalize_sequence(series, reference_point[col], scale_factor[col], inplace=True)

    # Check dataset type and process accordingly
    if isinstance(dataset, pd.DataFrame):
        for col in columns:
            if col in dataset.columns:
                dataset[col], reference_point[col], scale_factor[col] = normalise_single_column(dataset[col], col)

    elif isinstance(dataset, np.ndarray):
        for i, col in enumerate(columns):  # type: ignore
            # Flatten if more than 2 dimensions
            original_shape = dataset[..., i].shape
            values = dataset[..., i].flatten()
            values, reference_point[col], scale_factor[col] = normalise_single_column(values, col)
            dataset[..., i] = values.reshape(original_shape)
    elif isinstance(dataset, Sequence):  # Sequence of np.arrays
        for arr in dataset:
            for j, col in enumerate(columns):  # type: ignore
                arr[..., j], reference_point[col], scale_factor[col] = normalise_single_column(arr[..., j], col)
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    # Reference and scale factor have been filled
    typed_reference_point = cast(DR, reference_point)
    typed_scale_factor = cast(DR, scale_factor)

    return dataset, typed_reference_point, typed_scale_factor


@validate_call(config={"arbitrary_types_allowed": True})
def denormalise(
        dataset: T_D,
        ref: R,
        scale: R,
        columns: C = SPATIAL_COLUMNS,
        inplace: bool = False
) -> T_D:
    """
    Denormalises a dataset of trajectories or points using the provided reference points and scale factors,
    assuming the data was normalised using min-max scaling.

    :param dataset: Dataset to denormalise. Shape can be:
        - (n_trajectories, n_points, n_features) for trajectories
        - (n_points, n_features) for points
        - Sequence[np.ndarray], each element with shape (n_points, n_features)
        - pd.DataFrame with columns column names as provided in 'columns'
    :param ref: Reference points as Sequence/dict. If dict, keys are column names. If Sequence, in order of 'columns'.
        Example: {'lat': 0.0, 'lon': 0.0}
    :param scale: Scale factors as Sequence/dict. If dict, keys are column names. If Sequence, in order of 'columns'.
        Example: {'lat': 1.0, 'lon': 1.0}
    :param columns: Sequence of columns to denormalise. If 'dataset' is a DataFrame, use column names. If ndarray, use
        keys of 'ref' and 'scale' in order of columns.
    :param inplace: If True, modify dataset in place.
    :return: Denormalised dataset.
    """
    # Assert that columns math the keys of ref and scale
    if (isinstance(ref, dict) and set(ref.keys()) != set(columns)) or (
            isinstance(ref, Sequence) and not isinstance(columns[0], int)) or \
            (isinstance(scale, dict) and set(scale.keys()) != set(columns)) or (
            isinstance(scale, Sequence) and not isinstance(columns[0], int)):
        raise ValueError("Columns must match keys of ref and scale.")

    # Initialise reference and scale as dictionaries if provided as Sequences
    reference_point = cast(DR, make_dict(ref, columns))
    scale_factor = cast(DR, make_dict(scale, columns))
    del ref, scale

    # Make float it not already
    if isinstance(dataset, Sequence):
        dataset = [arr.astype(float, copy=(not inplace)) for arr in dataset]
    else:
        dataset = dataset.astype(float, copy=(not inplace))

    # Define helper to denormalise sequences (inplace = True because we already made a copy)
    def denormalise_single_column(series: S, col: Union[int, str]) -> S:
        return denormalize_sequence(series, reference_point[col], scale_factor[col], inplace=True)

    # Check dataset type and process accordingly
    if isinstance(dataset, pd.DataFrame):
        for col in columns:
            if col in dataset.columns:
                dataset[col] = denormalise_single_column(dataset[col], col)
    elif isinstance(dataset, np.ndarray):
        for i, col in enumerate(columns):  # type: ignore
            # Flatten if more than 2 dimensions
            original_shape = dataset[..., i].shape
            values = dataset[..., i].flatten()
            dataset[..., i] = denormalise_single_column(values, col).reshape(original_shape)
    elif isinstance(dataset, Sequence):  # Sequence of np.arrays
        for arr in dataset:
            for j, col in enumerate(columns):  # type: ignore
                arr[..., j] = denormalise_single_column(arr[..., j], col)
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    return dataset


@validate_call(config={"arbitrary_types_allowed": True})
def count_outside_bbox(
        data: Union[np.ndarray, Sequence[np.ndarray]],
        bbox: Dict[str, float],
        lon_first: bool = True
) -> int:
    """
    Count how many points in the dataset are outside the bounding box.

    :param data: Dataset to check. Shape can be:
        - (n_trajectories, n_points, n_features) for trajectories
        - (n_points, n_features) for points
        - Sequence[np.ndarray], each element with shape (n_points, n_features)
    :param bbox: Bounding box as dict with keys 'lon_min', 'lon_max', 'lat_min', 'lat_max'
    :param lon_first: If True, the first feature is longitude, else latitude
    """

    def num_outside(points: np.ndarray) -> int:
        if not lon_first:
            points = points[..., ::-1]
        return np.sum(
            (points[..., 0] < bbox['lon_min']) | (points[..., 0] > bbox['lon_max']) |
            (points[..., 1] < bbox['lat_min']) | (points[..., 1] > bbox['lat_max'])
        )  # type: ignore

    # Check dataset type and process accordingly
    count = 0
    if isinstance(data, np.ndarray):
        count = num_outside(data)
    elif isinstance(data, Sequence):  # Sequence of np.arrays
        for arr in data:
            count += num_outside(arr)
    else:
        raise ValueError(f"Unsupported dataset type: {type(data)}")

    return count
