#!/usr/bin/env python3
"""Definition of Utility Metrics."""
from typing import Tuple, Union

import numpy as np
from haversine import haversine, Unit
from scipy.spatial.distance import euclidean

from privtrajgen.metrics import PointDistanceMetric


def _haversine(p1: Union[Tuple[float, float], np.ndarray], p2: Union[Tuple[float, float], np.ndarray],
               lon_first: bool = True) -> float:
    """
    Compute the haversine distance between two points in meters.

    :param p1: Point 1 with shape (2,) = (longitude, latitude)
    :param p2: Point 2 with shape (2,) = (longitude, latitude)
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :return: The computed haversine distance between the two points.
    """
    if lon_first:
        p1 = (p1[1], p1[0])
        p2 = (p2[1], p2[0])
    return haversine(p1, p2, unit=Unit.METERS)


point_distance_metrics = {
    # All metrics expect (lon, lat) format as input!
    PointDistanceMetric.EUCLIDEAN: euclidean,
    PointDistanceMetric.HAVERSINE: _haversine,
}
