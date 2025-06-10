#!/usr/bin/env python3
"""Preprocessing Tools"""
from datetime import timedelta
from typing import Optional, List

import math
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

from privtrajgen.config import LAT, LON, TIMESTAMP, TID


def drop_out_of_bounds(
        df: pd.DataFrame,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        lat_label: str = LAT,
        lon_label: str = LON,
        inplace: bool = True
) -> pd.DataFrame:
    """
    Drop all points with longitude or latitude outside the given bounding box.
    :param df: Trajectory or set of locations.
    :param lon_min: Minimal longitude
    :param lon_max: Maximal longitude
    :param lat_min: Minimal latitude
    :param lat_max: Maximal latitude
    :param lat_label: Label of latitude column
    :param lon_label: Label of longitude column
    :param inplace: Modify the DataFrame in place
    :return:
    """
    res = df.drop(df[
                      (df[lon_label] < lon_min) | (df[lon_label] > lon_max) | (
                              df[lat_label] < lat_min) | (df[lat_label] > lat_max)
                      ].index, inplace=inplace)
    df = df if inplace else res
    res = df.reset_index(drop=True, inplace=inplace)
    df = df if inplace else res
    assert df is not None, "Error with inplace operations"
    return df


def resample_trajectory(t: pd.DataFrame, interval: int = 5, time: str = TIMESTAMP, uid: str = 'uid') -> pd.DataFrame:
    """
    Resamples a trajectory to a fixed interval.
    :param t: Trajectory as a DataFrame
    :param interval: Interval in seconds
    :param time: Label of the time column
    :param uid: Label of the uid column
    :return:
    """
    # Return empty DataFrame if trajectory is empty
    if len(t) == 0:
        return t
    # Convert to datetime
    if t[time].dtype != np.dtype('datetime64[ns]'):
        t[time] = pd.to_datetime(t[time])
    columns = t.columns
    # Resample
    t = t.resample(f'{interval}s', on=time)
    # Aggregate all other numerical columns with 'mean', and all other columns with 'first'
    aggregation_dic = {c: 'mean' if is_numeric_dtype(t[c]) else 'first' for c in columns}
    aggregation_dic[time] = 'mean'
    aggregation_dic[uid] = 'first'
    t = t.agg(aggregation_dic)
    # Drop NaNs
    t = t.dropna()
    # Reset index: drop=False to keep the timestamp as a column
    t = t.reset_index(drop=True)
    return t


def split_based_on_timediff(df: pd.DataFrame, interval: float, time_column: str = TIMESTAMP) -> List[pd.DataFrame]:
    """
    Split one trajectory into a list of shorter trajectories based on the gap between data points.

    :param df: The trajectory to split.
    :param interval: [SECONDS] The maximal time between to data points within a trajectory.
    :param time_column: Name of the time column
    :return: List of trajectories, each represented as pandas DataFrame.
    """
    interval = timedelta(seconds=interval)
    df = df.sort_values(by=time_column)
    df.reset_index(drop=True, inplace=True)
    df['tdiff'] = df[time_column].diff()
    split_indices: pd.DataFrame = df.loc[df['tdiff'] > interval]
    splits = [0] + list(split_indices.index) + [len(df)]
    dfs = [df.iloc[splits[i]: splits[i + 1]].drop(columns=['tdiff'], errors='ignore') for i in range(len(splits) - 1)]
    return dfs


def split_based_on_length(df: pd.DataFrame, max_len: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Split one trajectory into a list of shorter trajectories based on a maximal length.
    :param df: The trajectory to split (as a DataFrame)
    :param max_len: The maximal length of a trajectory. None means no splitting.
    :return: List of trajectories, each represented as pandas DataFrame.
                Even a list if only one trajectory is returned.
    """
    if len(df) == 0:
        # Empty DataFrame
        return []
    if max_len is None:
        # No splitting
        return [df]
    df.reset_index(drop=True, inplace=True)
    segments = math.ceil(len(df) / max_len)
    splits = list(range(0, max_len * segments + 1, max_len))
    dfs = [df.iloc[splits[i]:splits[i + 1]].reset_index(drop=True) for i in range(len(splits) - 1)]
    return dfs


def split_and_rename(df: pd.DataFrame, interval: Optional[float] = None, max_len: Optional[int] = None,
                     time_column: str = TIMESTAMP) -> List[pd.DataFrame]:
    """
    Split one trajectory into a list of shorter trajectories based on a maximal length and a maximal time interval, and
    rename the trajectories to unique IDs.
    :param df: The trajectory to split (as a DataFrame)
    :param interval: [SECONDS] Maximal time interval between two points. If None, no splitting
                        based on time is performed.
    :param max_len: Maximal length of a trajectory. If None, no splitting based on length is performed.
    :param time_column: Name of the time column
    :return: List of trajectories, each represented as pandas DataFrame.
                Even a list if only one trajectory is returned.
    """
    if len(df) == 0:
        return []
    tid = df[TID][0]
    if interval is not None:
        tmp = split_based_on_timediff(df, interval=interval, time_column=time_column)
    else:
        tmp = [df]
    trajs = []
    for t in tmp:
        trajs.extend(split_based_on_length(t, max_len=max_len))
    # Add unique ID
    for i, t in enumerate(trajs):
        new_tid = f"{tid}_{i}"
        t[TID] = new_tid
    return trajs


# noinspection PyUnusedLocal
def drop_duplicate_points(df: pd.DataFrame,
                          lat_label: str = LAT,
                          lon_label: str = LON,
                          time_label: str = TIMESTAMP,
                          inplace: bool = True
                          ) -> pd.DataFrame:
    """
    Remove points with duplicate timestamps. Choose the closer location to the surrounding points
    :param df: The DataFrame to modify (in place)
    :param lat_label: Label of latitude column
    :param lon_label: Label of longitude column
    :param time_label: Label of time column
    :param inplace: Modify the DataFrame in place
    :return: Return the modified DataFrame, but input was modified in-place
    """
    raise DeprecationWarning("We prefer to resample trajectories instead of dropping duplicate points. "
                             "Duplicate points are not are then removed implicitly.")
    columns = [lat_label, lon_label]  # type: ignore
    res = df.sort_values(by=time_label, inplace=inplace)
    df = df if inplace else res
    df.reset_index(drop=True, inplace=True)

    df['tdiff'] = df[time_label].diff()
    while len(df[(df.tdiff == timedelta(seconds=0))]) > 0:
        df['lon_diff'] = df[lon_label].diff()
        df['lat_diff'] = df[lat_label].diff()
        duplicates = list(df[(df.tdiff == timedelta(seconds=0)) & (
                df.lon_diff == 0) & (df.lat_diff == 0)].index)
        indices = df[(df.tdiff == timedelta(seconds=0)) & (
                (df.lon_diff != 0) | (df.lat_diff != 0))].index
        for i in indices:
            if i == len(df) - 1:
                # If the last point is an outlier, remove it
                duplicates.append(i)
                continue
            prev = df.iloc[i - 2][columns]
            p1 = df.iloc[i - 1][columns]
            p2 = df.iloc[i][columns]
            next_p = df.iloc[i + 1][columns]
            dist_p1 = np.linalg.norm(p1 - prev) + np.linalg.norm(next_p - p1)
            dist_p2 = np.linalg.norm(p2 - prev) + np.linalg.norm(next_p - p2)
            if dist_p1 > dist_p2:
                duplicates.append(i - 1)
            else:
                duplicates.append(i)

        df.drop(duplicates, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['tdiff'] = df[time_label].diff()

    df.drop(columns=['tdiff', 'lon_diff', 'lat_diff'], inplace=True, errors='ignore')

    return df
