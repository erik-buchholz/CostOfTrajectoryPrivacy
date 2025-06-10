#!/usr/bin/env python3
""" """
from typing import Dict, Tuple

import numpy as np


def divide_grids(boundary: Dict[str, float], grids_num: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide the boundary into grids_num grids.
    DiffTraj uses a 12x12 grids by default.
    :param boundary: Dictionary containing the boundary of the area as follows:
        {'lat_min': float, 'lat_max': float, 'lon_min': float, 'lon_max': float}
    :param grids_num: Dimension of the grid
    :return: (latgrids, longrids) arrays containing the grid boundaries. (including outer boundaries, i.e.,
                len(latgrids) = len(longrids) = grids_num + 1)
    """
    lat_min, lat_max = boundary['lat_min'], boundary['lat_max']
    lon_min, lon_max = boundary['lon_min'], boundary['lon_max']
    # Create arrays of latitude and longitude values. Include the outer boundaries.
    latgrids = np.linspace(lat_min, lat_max, grids_num + 1)
    longrids = np.linspace(lon_min, lon_max, grids_num + 1)
    return latgrids, longrids


def compute_grid_id(lon_idx: int, lat_idx: int, n_lon_cols: int) -> int:
    """
    Compute the grid ID from the longitude and latitude indices.
    :param lon_idx: Longitude index (column of the grid)
    :param lat_idx: Latitude index (row of the grid)
    :param n_lon_cols: Number of longitude columns in the grid (Note: n_lon_cols = len(longrids) - 1)
    :return: Grid ID ∈ [0, (n_lat_rows) * (n_lon_cols) - 1]
    """
    return lat_idx * n_lon_cols + lon_idx


def compute_grid_cell(lon: float, lat: float, longrids: np.ndarray, latgrids: np.ndarray,
                      check: bool = True) -> int:
    """
    Compute the grid cell for a given latitude and longitude.
    :param lon: Longitude (column of the grid)
    :param lat: Latitude (row of the grid)
    :param longrids: Array of longitude grid points (including outer boundaries)
    :param latgrids: Array of latitude grid points (including outer boundaries)
    :param check: Whether to check if the point is within the boundary
    :return: Grid ID ∈ [0, (len(latgrids) - 1) * (len(longrids) - 1) - 1]

    Example Grid Layout:
              |lon_0|lon_1|lon_2|lon_3|
              |_____|_____|_____|_____|
        lat_0 |  0  |  1  |  2  |  3  |
        lat_1 |  4  |  5  |  6  |  7  |
        lat_2 |  8  |  9  | 10  | 11  |
        lat_3 | 12  | 13  | 14  | 15  |
    """
    # Check if the point is within the boundary
    if check and (lon < longrids[0] or lon > longrids[-1] or lat < latgrids[0] or lat > latgrids[-1]):
        raise ValueError("Point is outside the boundary.")
    # [:-1] -> Remove upper boundary so that the upper boundary itself is included
    # max(..., 0) -> Ensure that the index is not smaller than 0 even if the point is outside the boundary
    lat_idx = max(np.searchsorted(latgrids[:-1], lat, side='right') - 1, 0)
    lon_idx = max(np.searchsorted(longrids[:-1], lon, side='right') - 1, 0)
    # (len(longrids)-1) to exclude the upper boundary
    return compute_grid_id(lon_idx=lon_idx, lat_idx=lat_idx, n_lon_cols=len(longrids) - 1)  # type: ignore
