#!/usr/bin/env python3
"""Compute DiffTraj conditional information.
Refer to https://github.com/Yasoz/DiffTraj/issues/10 for more information.

1. Departure Time: _Categorical_ Attribute of the time of day in 5-minute steps. _Not Normalised_
    288 possible values.
2. Total Distance: _Numerical_ Attribute normalised via mean and standard deviation (see above).
    This is the sum of the distances between all two consecutive points within the trajectory
3. Total Duration: _Numerical_ Attribute normalised via mean and standard deviation (see above).
    This is total duration in seconds.
4. Total Length: _Numerical_ Attribute normalised via mean and standard deviation (see above).
    Number of points in the trajectory.
5. Avg Distance: _Numerical_ Attribute normalised via mean and standard deviation (see above).
    Average distance (in m) between 2 consecutive locations.
6. Average Speed: _Numerical_ Attribute normalised via mean and standard deviation (see above).
    `Total Length / Total Duration`
7. Start Grid ID: _Categorical_ Attribute. _Not Normalised_ Values from $0 - 140$
   Used `divide_grids` from utils.py.
8. End Grid ID: _Categorical_ Attribute. _Not Normalised_
   Same as 7.
"""
from typing import List, Dict, Union, Tuple, Sequence

import numpy as np
import pandas as pd
from haversine import haversine_vector

from .constants import COND_ATTRIBUTES_DiffTraj, NORMALISED_ATTRIBUTES_DiffTraj
from privtrajgen.config.constants import *
from privtrajgen.utils.grid import divide_grids, compute_grid_cell


def normalise_conditional_info(conditional_info: Dict[str, float], mean: Dict[str, float],
                               std: Dict[str, float],
                               normalised_attributes: Sequence[str] = NORMALISED_ATTRIBUTES_DiffTraj,
                               ordered: bool = True, order: List[str] = COND_ATTRIBUTES_DiffTraj) -> Union[Dict[str, float], List[float]]:
    """
    Normalise the conditional information using the mean and standard deviation.
    :param conditional_info: Dictionary of conditional information
    :param mean: Mean of the conditional information
    :param std: Standard deviation of the conditional information
    :param normalised_attributes: List of attributes to normalise
    :param ordered: If True, return the normalised attributes in the order of `order`
    :param order: Order of the attributes
    :return: Normalised conditional information
    """
    for attr in normalised_attributes:
        conditional_info[attr] = (conditional_info[attr] - mean[attr]) / std[attr]
    if ordered:
        return [conditional_info[attr] for attr in order]
    return conditional_info


def compute_conditional_info(traj: pd.DataFrame,
                             bbox: Dict[str, float],
                             lat: str = LAT,
                             lon: str = LON,
                             timestamp: str = TIMESTAMP,
                             hour: str = HOUR,
                             dow: str = 'day',
                             categorical_timestamp: bool = False,
                             order: List[str] = COND_ATTRIBUTES_DiffTraj,
                             as_dict: bool = False,
                             grid_dim: int = 12
                             ) -> Union[Dict[str, float], List[float]]:
    """
    Compute the conditional information for a trajectory.
    :param traj: Trajectory DataFrame
    :param bbox: Boundary of the area as follows:
        {'lat_min': float, 'lat_max': float, 'lon_min': float, 'lon_max': float}
    :param lat: Name of the latitude column
    :param lon: Name of the longitude column
    :param timestamp: Name of the timestamp column [ignored if categorical_timestamp is True]
    :param hour: Name of the hour column [ignored if categorical_timestamp is False]
    :param dow: Name of the day of the week column [ignored if categorical_timestamp is False]
    :param categorical_timestamp: If True, timestamp provided as hour and day of the week.
    :param order: Order of the attributes to return
    :param as_dict: If True, return as dictionary
    :param grid_dim: Number of cells per axis for the grid
    :return:
    """
    conditional_info: Dict[str, Union[int, float]] = dict()

    # Departure Time
    conditional_info['dep_time'] = compute_departure_time(traj, hour, timestamp, categorical_timestamp)
    # Total Distance
    conditional_info['total_dist'], conditional_info['avg_dist'] = compute_total_and_avg_distance(traj[[lat, lon]])
    # Total Duration
    conditional_info['total_dur'] = compute_total_duration(
        traj, timestamp=timestamp, categorical_timestamp=categorical_timestamp, hour=hour, dow=dow
    )
    # Total Length
    conditional_info['len'] = compute_total_length(traj)
    # Avg Speed
    conditional_info['avg_speed'] = compute_avg_speed(conditional_info['total_dist'], conditional_info['total_dur'])
    # Compute the grid cell boundaries
    latgrids, longrids = divide_grids(bbox, grids_num=grid_dim)
    # Start Grid ID
    conditional_info['sid'] = compute_grid_cell(lat=traj[lat].iloc[0], lon=traj[lon].iloc[0], latgrids=latgrids, longrids=longrids)
    # End Grid ID
    conditional_info['eid'] = compute_grid_cell(lat=traj[lat].iloc[-1], lon=traj[lon].iloc[-1], latgrids=latgrids, longrids=longrids)
    if as_dict:
        return conditional_info
    return [conditional_info[attr] for attr in order]


def compute_departure_time(traj: pd.DataFrame, hour: str = HOUR, timestamp: str = TIMESTAMP,
                           categorical_timestamp: bool = False) -> int:
    if categorical_timestamp:
        hour_value = traj[hour].iloc[0]
        # dow = traj[dow].iloc[0]  # Not used
        dep_time = compute_departure_time_as_category(hour_value, minute=0)
    else:
        dt: pd.Timestamp = traj[timestamp].iloc[0]
        dep_time = compute_departure_time_as_category(dt.hour, dt.minute)
    return dep_time

def compute_departure_time_as_category(hour: int, minute: int, step: int = 5) -> int:
    """Categorical_ Attribute of the time of day in X-minute steps. (Round down to the nearest step)
    --> 288 possible values.
    Not normalized.
    """
    value = hour * 12 + minute // step
    return value


def compute_total_and_avg_distance(lat_lon: np.ndarray) -> Tuple[float, float]:
    """
    Compute the total and average distance between two consecutive points in a trajectory.
    :param lat_lon: Numpy array of shape (n_points, 2) with each point being *(lat, lon)*
    :return: total distance, average distance
    """
    # Use haversine distance
    dists = haversine_vector(lat_lon[:-1], lat_lon[1:], unit='m')
    total_dist = np.sum(dists)
    avg_dist = np.mean(dists)
    return total_dist, avg_dist


def compute_total_duration(traj: pd.DataFrame, timestamp: str = TIMESTAMP, categorical_timestamp: bool = False,
                           hour: str = HOUR, dow: str = DOW) -> float:
    """
    Compute the total duration of a trajectory.
    :param traj: Trajectory DataFrame
    :param timestamp: Column name of the timestamp [ignored if categorical_timestamp is True]
    :param categorical_timestamp: If True, timestamp provided as hour and day of the week.
    :param hour: Column name of the hour [ignored if categorical_timestamp is False]
    :param dow: Column name of the day of the week [ignored if categorical_timestamp is False]
    :return: Total Duration in seconds
    """
    if categorical_timestamp:
        start_time = traj[hour].iloc[0] * 3600 + traj[dow].iloc[0] * 86400
        end_time = traj[hour].iloc[-1] * 3600 + traj[dow].iloc[-1] * 86400
        total_dur = end_time - start_time
    else:
        timestamps = traj[timestamp]
        total_dur = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    return total_dur

def compute_total_length(traj: pd.DataFrame) -> int:
    """
    Compute the total length of a trajectory.
    :param traj: Trajectory DataFrame
    :return: Total Length
    """
    return len(traj)

def compute_avg_speed(total_dist: float, total_dur: float) -> float:
    """
    Compute the average speed of a trajectory in m/s.
    [https://github.com/Yasoz/DiffTraj/issues/10#issuecomment-2151257183]
    :param total_dist: Total distance in meters
    :param total_dur: Total duration in seconds
    :return: Average Speed in m/s
    """
    # Note that duration might be 0 in case of categorical timestamps
    speed = 0 if total_dur == 0 else total_dist / total_dur
    return speed

