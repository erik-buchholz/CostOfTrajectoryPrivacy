#!/usr/bin/env python3
"""
Trajectory-level Preservation Utility Metrics.

These metrics require a 1:1 mapping between trajectories from the compared datasets.
"""
import pickle
from typing import List, Tuple, Callable, Union, cast, Dict, Optional, Any

import numpy as np
from dtw import dtw
from haversine import haversine_vector, Unit
from pydantic import validate_call
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff

from privtrajgen.metrics.distances import _haversine
from privtrajgen.metrics.keys import *
from privtrajgen.metrics.utils import verify_batch_sizes


@validate_call(config={"arbitrary_types_allowed": True})
def _orientation_to_metric_kwargs(lon_first: bool, metric_kwargs: dict) -> dict:
    """
    Set the orientation of the trajectories in the metric_kwargs if not already set.
    Raises an error if the orientation is set and different from the provided lon_first.

    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :param metric_kwargs: Additional keyword arguments to pass to the distance metric function.
    :return: The updated metric_kwargs.
    """
    if 'lon_first' in metric_kwargs:
        if metric_kwargs['lon_first'] != lon_first:
            raise ValueError(
                "The orientation of the trajectories in the metric_kwargs is different from the provided lon_first.")
    else:
        metric_kwargs['lon_first'] = lon_first
    return metric_kwargs


@validate_call(config={"arbitrary_types_allowed": True})
def hausdorff_directed_haversine(t1: np.ndarray, t2: np.ndarray, lon_first: bool = True) -> Tuple[float, float]:
    """
    Compute the directed Hausdorff distance of 2 trajectories, based on longitude and latitude.

    :param t1: Trajectory 1 with shape (n_points, 2)
    :param t2: Trajectory 2 with shape (n_points, 2)
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :return: The computed directed Hausdorff distance (in _meter_)
    """
    if len(t1) == 0 or len(t2) == 0:
        raise ValueError("Trajectories must not be empty.")
    if lon_first:
        # Haversine formula requires (lat, lon) format
        t1 = t1[:, ::-1]
        t2 = t2[:, ::-1]
    # Compute the distances using haversine formula
    distances = haversine_vector(t1, t2, Unit.METERS, comb=True)

    # Compute the directed Hausdorff distance
    direction1 = np.max(np.min(distances, axis=1))
    direction2 = np.max(np.min(distances, axis=0))

    return direction1, direction2


@validate_call(config={"arbitrary_types_allowed": True})
def hausdorff_distance(t1: np.ndarray, t2: np.ndarray, use_haversine: bool = True,
                       lon_first: bool = True) -> float:
    """
        Compute the Hausdorff distance of 2 trajectories, based on longitude and latitude or x and y.

        :param t1: Trajectory 1 with shape (n_points, 2)
        :param t2: Trajectory 2 with shape (n_points, 2)
        :param use_haversine: Use haversine distance instead of Euclidean distance
        :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
        :return: The computed Hausdorff distance (in _meter_ if using haversine)
        """
    if len(t1) == 0 or len(t2) == 0 or len(t1) != len(t2):
        raise ValueError("Trajectories must not be empty and have the same length.")
    if use_haversine:
        # Compute the distance using haversine formula
        direction1, direction2 = hausdorff_directed_haversine(t1, t2, lon_first=lon_first)
    else:
        direction1 = directed_hausdorff(t1, t2)[0]
        direction2 = directed_hausdorff(t2, t1)[0]

    return max(direction1, direction2)


@validate_call(config={"arbitrary_types_allowed": True})
def euclidean_distance(t1: np.ndarray, t2: np.ndarray, normalise: bool = False, lon_first: bool = True) -> float:
    """
    Compute the Euclidean distance between two trajectories.

    There are two ways one could define the Euclidean distance between two trajectories:
    1) The sum of the Euclidean distances between corresponding points in the two trajectories.
        --> np.sum(np.linalg.norm(t1 - t2, axis=1)) [/ len(t1)]
        --> $$\\text{Euclidean Distance}(t_1, t_2) = \\frac{\\sum_{i=1}^{n}\\sqrt{(x_i - u_i)^2 + (y_i - v_i)^2)}}{n}$$
        - where [/len(t1)] is optional to normalize the distance by the number of points.
    2) The Euclidean distance between the two trajectories as a whole.
        --> np.linalg.norm(t1 - t2)
        --> $$\\text{Euclidean Distance}(t_1, t_2) = \\sqrt{\\sum_{i=1}^{n}((x_i - u_i)^2 + (y_i - v_i)^2)}$$

    RAoPT [2] use definition (1), while the SDD paper [1] uses definition (2).
    We will use definition (1), as it is also proposed int the survey by Su et al. [3].

    [1] K. Jiang, D. Shao, S. Bressan, T. Kister, and K.-L. Tan,
    “Publishing trajectories with differential privacy guarantees,”
    in Proceedings of the 25th International Conference on Scientific and Statistical Database Management - SSDBM,
    New York, New York, USA: ACM Press, 2013, p. 1. doi: 10.1145/2484838.2484846.
    [2] E. Buchholz, A. Abuadbba, S. Wang, S. Nepal, and S. S. Kanhere,
    “Reconstruction Attack on Differential Private Trajectory Protection Mechanisms,”
    in Proceedings of the 38th Annual Computer Security Applications Conference, in ACSAC ’22.
    New York, NY, USA: ACM, Dec. 2022, pp. 279–292. doi: 10.1145/3564625.3564628.
    [3] H. Su, S. Liu, B. Zheng, X. Zhou, and K. Zheng,
    “A survey of trajectory distance measures and performance evaluation,”
    The VLDB Journal, vol. 29, no. 1, pp. 3–32, Jan. 2020, doi: 10.1007/s00778-019-00574-9.



    :param t1: Trajectory 1 with shape (n_points, 2)
    :param t2: Trajectory 2 with shape (n_points, 2)
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
        For the Euclidean distance, the order of the coordinates does not matter. Just for compatibility with
        the other distance metrics.
    :param normalise: Normalize the distance by the number of points
    :return: The computed Euclidean distance between the two trajectories.
    """
    if lon_first:
        pass
    if len(t1) == 0 or len(t2) == 0 or len(t1) != len(t2):
        raise ValueError("Trajectories must not be empty and have the same length.")
    distance = np.sum(np.linalg.norm(t1 - t2, axis=1))  # Variant 1
    if normalise:
        distance /= len(t1)
    return distance
    # return np.linalg.norm(t1 - t2)  # Variant 2


@validate_call(config={"arbitrary_types_allowed": True})
def normalised_euclidean_distance(t1: np.ndarray, t2: np.ndarray, lon_first: bool = True) -> float:
    """
    Compute the normalised Euclidean distance between two trajectories.

    The normalised Euclidean distance is the Euclidean distance between the two trajectories as a whole,
    normalised by the number of points in the trajectories.

    :param t1: Trajectory 1 with shape (n_points, 2)
    :param t2: Trajectory 2 with shape (n_points, 2)
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
        For the Euclidean distance, the order of the coordinates does not matter. Just for compatibility with
        the other distance metrics.
    :return: The computed normalised Euclidean distance between the two trajectories.
    """
    return euclidean_distance(t1, t2, normalise=True, lon_first=lon_first)


@validate_call(config={"arbitrary_types_allowed": True})
def haversine_distance(t1: np.ndarray, t2: np.ndarray, normalise: bool = False, lon_first: bool = True) -> float:
    """
    Compute the haversine distance between two trajectories in meters.

    :param t1: Trajectory 1 with shape (n_points, 2 = (latitude, longitude))
    :param t2: Trajectory 2 with shape (n_points, 2 = (latitude, longitude))
    :param normalise: Normalize the distance by the number of points
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :return: The computed haversine distance between the two trajectories.
    """
    if len(t1) == 0 or len(t2) == 0 or len(t1) != len(t2):
        raise ValueError("Trajectories must not be empty and have the same length.")
    if lon_first:
        # Haversine formula requires (lat, lon) format
        t1 = t1[:, ::-1]
        t2 = t2[:, ::-1]
    distance = haversine_vector(t1, t2, Unit.METERS).sum()
    if normalise:
        distance /= len(t1)
    return distance


@validate_call(config={"arbitrary_types_allowed": True})
def normalised_haversine_distance(t1: np.ndarray, t2: np.ndarray, lon_first: bool = True) -> float:
    """
    Compute the normalised haversine distance between two trajectories in meters.

    The normalised haversine distance is the haversine distance between the two trajectories as a whole,
    normalised by the number of points in the trajectories.

    :param t1: Trajectory 1 with shape (n_points, 2)
    :param t2: Trajectory 2 with shape (n_points, 2)
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :return: The computed normalised haversine distance between the two trajectories.
    """
    return haversine_distance(t1, t2, normalise=True, lon_first=lon_first)


@validate_call(config={"arbitrary_types_allowed": True})
def dtw_distance(t1: np.ndarray, t2: np.ndarray, use_haversine: bool = False,
                 return_alignment: bool = False, lon_first: bool = True) -> Union[float, Tuple[float, list]]:
    """
    Compute the Dynamic Time Warping (DTW) distance of 2 trajectories.

    :param t1: Trajectory 1 with shape (n_points, 2)
    :param t2: Trajectory 2 with shape (n_points, 2)
    :param use_haversine: Use haversine distance instead of Euclidean distance
    :param return_alignment: If True, return the alignment
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :return: The computed DTW distance between the two trajectories, and optionally the alignment.
    """
    if len(t1) == 0 or len(t2) == 0:
        raise ValueError("Trajectories must not be empty.")

    metric: Union[str, Callable]
    if use_haversine:
        if not lon_first:
            # _haversine expects (lon, lat) format
            t1 = t1[:, ::-1]
            t2 = t2[:, ::-1]
        # Compute the distance using haversine formula
        metric = _haversine
    else:
        # metric = euclidean  # fastdtw
        metric = 'euclidean'  # dtw-python

    # Compute the DTW distance between the two trajectories using fastdtw
    # Approximated by default, but accurate for radius=max(len(t1), len(t2))
    # distance, alignment = fastdtw(t1, t2, dist=metric, radius=max(len(t1), len(t2)))

    # Compute the DTW distance between the two trajectories using dtw-python as it's accurate and faster
    # We use step_pattern='symmetric1' as it's simpler and the same as used in fastdtw, used by related work
    # https://github.com/WenMellors/TS-TrajGen/blob/master/utils/evaluate_funcs.py
    result = dtw(t1, t2, dist_method=metric, distance_only=not return_alignment, keep_internals=False,
                 step_pattern='symmetric1')
    distance = result.distance
    alignment = list(zip(result.index1, result.index2)) if return_alignment else None

    if return_alignment:
        return distance, alignment  # type: ignore

    return distance


@validate_call(config={"arbitrary_types_allowed": True})
def batched_trajectory_distance(batch1: List[np.ndarray] | np.ndarray, batch2: List[np.ndarray] | np.ndarray,
                                distance_metric: Union[TrajectoryDistanceMetric, Callable], lon_first: bool = True,
                                **metric_kwargs: Any,) -> np.ndarray:
    """
    Compute the distance for two ordered batches of trajectories using a specified distance metric,
    where each trajectory in batch1 corresponds to a trajectory in batch2.

    :param batch1: List of trajectories from the first batch, each of shape (n_points, 2).
    :param batch2: List of trajectories from the second batch, each of shape (n_points, 2).
    :param distance_metric: A callable that computes the distance between two trajectories or a string
                            representing the distance metric.
                            The callable should have the signature distance_metric(traj1, traj2, **metric_kwargs).
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :param metric_kwargs: Additional keyword arguments to pass to the distance_metric function.
    :return: A numpy array of distances, one for each pair of corresponding trajectories.
    """
    # Ensure the batches have the same length
    verify_batch_sizes(batch1, batch2)

    # Verify if lon_first already in metric_kwargs, set if not and raise error if values are different
    metric_kwargs = _orientation_to_metric_kwargs(lon_first, metric_kwargs)

    # Check if the distance metric is a string
    if isinstance(distance_metric, (TrajectoryDistanceMetric, str)):
        distance_metric = TrajectoryDistanceMetrics[distance_metric]
        distance_metric = cast(Callable, distance_metric)

    # Compute the distance for each corresponding pair
    distances = np.zeros(len(batch1))

    for i in range(len(batch1)):
        distances[i] = distance_metric(batch1[i], batch2[i], **metric_kwargs)

    return distances


@validate_call(config={"arbitrary_types_allowed": True})
def is_picklable(obj: object) -> bool:
    """Check if an object is picklable."""
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError):
        return False
    return True


@validate_call(config={"arbitrary_types_allowed": True})
def batched_trajectory_distance_minimized(batch1: List[np.ndarray] | np.ndarray, batch2: List[np.ndarray] | np.ndarray,
                                          distance_metric: Union[TrajectoryDistanceMetric, Callable],
                                          lon_first: bool = True,
                                          **metric_kwargs: Any,) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
    """
    Compute the distance for all possible combinations of trajectories from batch1 and batch2 using a specified
    distance metric, and match them such that the total distance is minimized.

    :param batch1: List of trajectories from the first batch, each of shape (n_points, 2).
    :param batch2: List of trajectories from the second batch, each of shape (n_points, 2).
    :param distance_metric: A callable that computes the distance between two trajectories or a string
                            representing the distance metric.
                            The callable should have the signature distance_metric(traj1, traj2, **metric_kwargs).
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :param metric_kwargs: Additional keyword arguments to pass to the distance_metric function.
    :return: (matching, minimized_distances, total_distance)
                - matching: A list of tuples, where each tuple contains the indices of the matched trajectories.
                - minimized_distances: The minimized distances for each pair of corresponding trajectories.
                - total_distance: The total distance of the optimal assignment.
    """
    # Ensure the batches have the same length
    verify_batch_sizes(batch1, batch2)

    # Verify if lon_first already in metric_kwargs, set if not and raise error if values are different
    metric_kwargs = _orientation_to_metric_kwargs(lon_first, metric_kwargs)

    n = len(batch1)

    # Check if the distance metric is a string
    if isinstance(distance_metric, str):
        distance_metric = TrajectoryDistanceMetrics[distance_metric]

    # Compute the distance matrix for all possible combinations
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = distance_metric(batch1[i], batch2[j], **metric_kwargs)

    # Find the optimal assignment using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Create the matching
    matching = list(zip(row_indices, col_indices))

    # Retrieve the minimized distances
    minimized_distances = distance_matrix[row_indices, col_indices]

    # Compute total distance of optimal assignment
    total_distance = minimized_distances.sum()

    return matching, minimized_distances, total_distance


@validate_call(config={"arbitrary_types_allowed": True})
def reorder_dataset_to_matching(dataset: List[np.ndarray] | np.ndarray, matching: List[tuple[int, int]]) -> List[
    np.ndarray]:
    """
    Reorder a dataset according to a matching.

    :param dataset: The dataset to reorder, a list of trajectories.
    :param matching: A list of tuples, where each tuple contains the indices of the matched trajectories:
        matching[i] = (idx1, idx2) --> dataset1[idx1] is matched with dataset2[idx2].
        I.e., the reordering is such that reorder_dataset[idx1] = dataset[idx2].
    :return: The reordered dataset.
    """
    if len(dataset) != len(matching):
        raise ValueError("Dataset and matching must have the same length.")
    reordered_dataset: List[Optional[np.ndarray]] = [None] * len(dataset)

    for idx1, idx2 in matching:
        reordered_dataset[idx1] = dataset[idx2]

    # Check for None entries
    if any(entry is None for entry in reordered_dataset):
        raise ValueError("Reordering failed. Some entries in the reordered dataset are None.")

    return reordered_dataset  # type: ignore


@validate_call(config={"arbitrary_types_allowed": True})
def match_trajectories(
        batch1: list[np.ndarray],
        batch2: list[np.ndarray],
        distance_metric: Union[TrajectoryDistanceMetric, Callable],
        lon_first: bool = True,
        **metric_kwargs: Any,
) -> Tuple[List[Tuple[int, int]], float]:
    """Create a 1:1 mapping between trajectories from two batches using a specified distance metric.

    :param batch1: List of trajectories from the first batch, each of shape (n_points, 2).
    :param batch2: List of trajectories from the second batch, each of shape (n_points, 2).
    :param distance_metric: A callable that computes the distance between two trajectories or a string
                            representing the distance metric.
                            The callable should have the signature distance_metric(traj1, traj2, **metric_kwargs).
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :param metric_kwargs: Additional keyword arguments to pass to the distance_metric function.
    :return: (matching, total_distance)
                - matching: A list of tuples, where each tuple contains the indices of the matched trajectories.
                - total_distance: The total distance of the optimal assignment.
    """
    # Ensure the batches have the same length
    verify_batch_sizes(batch1, batch2)

    # Verify if lon_first already in metric_kwargs, set if not and raise error if values are different
    metric_kwargs = _orientation_to_metric_kwargs(lon_first, metric_kwargs)

    matching, minimized_distances, total_distance = batched_trajectory_distance_minimized(
        batch1, batch2, distance_metric, **metric_kwargs)

    return matching, total_distance


@validate_call(config={"arbitrary_types_allowed": True})
def aggregate_metric(distances: np.ndarray | List[float]) -> dict:
    """
    Aggregate the distances of a metric.

    :param distances: The distances of a metric.
    :return: A dictionary containing the aggregated metric.
    """
    return {
        MEAN: np.mean(distances),
        STD: np.std(distances),
        NUM: len(distances),
        SUM: np.sum(distances)
    }


TrajectoryDistanceMetrics: Dict[TrajectoryDistanceMetric, Callable] = {
    TrajectoryDistanceMetric.HAUSDORFF: hausdorff_distance,
    TrajectoryDistanceMetric.DTW: dtw_distance,
    TrajectoryDistanceMetric.EUCLIDEAN: euclidean_distance,
    TrajectoryDistanceMetric.HAVERSINE: haversine_distance,
}
