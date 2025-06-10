#!/usr/bin/env python3
"""Trajectory-level Statistics Utility Metrics."""
from typing import Optional, Dict, List, Union, Callable, cast, Sequence, Any

import numpy as np
from pydantic import validate_call
from scipy.spatial.distance import cdist

from privtrajgen.config import JSD_MODE_STATISTICS, N_BINS_TRAJ_STATS, JSDModes
from privtrajgen.utils.grid import divide_grids, compute_grid_cell
from .distances import point_distance_metrics, PointDistanceMetric
from .distribution import JSD
from .keys import TTD, DIAMETER

CDIST_METRICS = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
                 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
                 'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
                 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                 'sokalsneath', 'sqeuclidean', 'yule']


@validate_call(config={"arbitrary_types_allowed": True})
def total_travelled_distance(traj: np.ndarray,
                             metric: Union[PointDistanceMetric, Callable] = PointDistanceMetric.EUCLIDEAN,
                             **metric_kwargs: Any,) -> float:
    """
    Compute the total travelled distance of a trajectory.

    :param traj: The trajectory to compute the TTD of with shape (n_samples, 2 = (longitude, latitude)).
    :param metric: The distance metric to use for computing the distance between points.
    :param metric_kwargs: Additional keyword arguments to pass to the distance metric function.
    :return: The total travelled distance of the trajectory.
    """
    if callable(metric):
        dist_func = metric
    else:
        dist_func = point_distance_metrics[metric]

    return np.sum([dist_func(traj[i], traj[i + 1], **metric_kwargs) for i in range(len(traj) - 1)])


@validate_call(config={"arbitrary_types_allowed": True})
def trajectory_diameter(traj: np.ndarray, metric: Union[PointDistanceMetric, Callable] = PointDistanceMetric.EUCLIDEAN,
                        **metric_kwargs: Any,) -> float:
    """
    Compute the diameter of a trajectory.

    The diameter of a trajectory is the maximum distance between any two points in the trajectory.

    :param traj: The trajectory to compute the diameter of with shape (n_samples, 2 = (longitude, latitude)).
    :param metric: The distance metric to use for computing the distance between points.
    :param metric_kwargs: Additional keyword arguments to pass to the distance metric function.
    :return: The diameter of the trajectory.
    """
    if callable(metric):
        dist_func = metric
    else:
        dist_func = point_distance_metrics[metric]

    return np.max(
        [dist_func(traj[i], traj[j], **metric_kwargs) for i in range(len(traj)) for j in range(i + 1, len(traj))])


@validate_call(config={"arbitrary_types_allowed": True})
def trajectory_statistics(traj: np.ndarray,
                          metric: Union[PointDistanceMetric, Callable] = PointDistanceMetric.EUCLIDEAN,
                          lon_first: bool = True, **metric_kwargs: Any, ) -> dict:
    """
    Compute statistics of a trajectory.
    Computing trajectory statistics separately is very inefficient,
    as it requires multiple passes over the trajectory.

    :param traj: The trajectory to compute statistics of with shape (n_samples, 2).
    :param metric: The distance metric to use for computing the distance between points.
    :param lon_first: If True, trajectory points are in the format (longitude, latitude) else (latitude, longitude).
    :param metric_kwargs: Additional keyword arguments to pass to the distance metric function.
    :return: A dictionary containing the computed statistics.
        - 'total_distance': The total travelled distance of the trajectory.
        - 'diameter': The diameter of the trajectory.
    """
    if callable(metric) or metric in CDIST_METRICS:
        dist_func = metric
    else:
        dist_func = point_distance_metrics[metric]

    if not lon_first:
        # We assume longitude is the first element and latitude is the second element
        traj = traj[:, ::-1]

    distances = cdist(traj, traj, metric=dist_func, **metric_kwargs)

    total_distance = np.sum(np.diagonal(distances, offset=1))
    diameter = np.max(distances)

    return {
        TTD: total_distance,
        DIAMETER: diameter,
    }


@validate_call(config={"arbitrary_types_allowed": True})
def trajectory_statistics_error(real_traj_dataset: Union[np.ndarray, Sequence[np.ndarray]],
                                gen_traj_dataset: Union[np.ndarray, Sequence[np.ndarray]],
                                num_bins: int = N_BINS_TRAJ_STATS,
                                metric: Union[PointDistanceMetric, Callable] = PointDistanceMetric.EUCLIDEAN,
                                lon_first: bool = True, jsd_mode: JSDModes = JSD_MODE_STATISTICS,
                                **metric_kwargs: Any,) -> Dict[str, float]:
    """
    Compute the error between the statistics of two trajectory datasets using the JSD.

    :param real_traj_dataset: A dataset of real trajectories with shape (n_trajectories, n_samples, 2).
    :param gen_traj_dataset: A dataset of generated trajectories with shape (n_trajectories, n_samples, 2).
    :param num_bins: Number of bins for the histogram in the trajectory statistics.
    :param metric: The distance metric to use for computing the distance between points.
    :param lon_first: If True, trajectory points are in the format (longitude, latitude) else (latitude, longitude).
    :param jsd_mode: The mode to use for the Jensen-Shannon Divergence (JSD) computation.
    :param metric_kwargs: Additional keyword arguments to pass to the distance metric function.
    :return: JSD(M(real), M(gen)) where M is the metric function.
    """
    # Compute the statistics of the real and generated datasets
    real_stats: List[dict] = [trajectory_statistics(traj, metric=metric, lon_first=lon_first, **metric_kwargs) for traj
                              in
                              real_traj_dataset]
    gen_stats: List[dict] = [trajectory_statistics(traj, metric=metric, lon_first=lon_first, **metric_kwargs) for traj
                             in gen_traj_dataset]

    # Compute the JSD between the statistics of the real and generated datasets
    statistics = list(real_stats[0].keys())

    # Initialise empty lists to collect values for each statistic
    data_real: Dict[str, List[float]] = {statistic: [] for statistic in statistics}
    data_gen: Dict[str, List[float]] = {statistic: [] for statistic in statistics}

    # Collect real and generated stats per trajectory
    for real_stat, gen_stat in zip(real_stats, gen_stats):
        for statistic in statistics:
            data_real[statistic].append(real_stat[statistic])
            data_gen[statistic].append(gen_stat[statistic])

    # Convert lists to np.arrays for each statistic
    data_real: Dict[str, np.ndarray] = {statistic: np.array(values) for statistic, values in data_real.items()}
    data_gen: Dict[str, np.ndarray] = {statistic: np.array(values) for statistic, values in data_gen.items()}

    # Compute the JSD between the statistics of the real and generated datasets
    results = {}
    for statistic in statistics:
        # Create the probability distributions for the real and generated datasets that can be compared via JSD
        dist_real = np.histogram(data_real[statistic], bins=num_bins, density=True)[0]
        dist_gen = np.histogram(data_gen[statistic], bins=num_bins, density=True)[0]
        results[statistic] = JSD(dist_real, dist_gen, mode=jsd_mode)

    return results


@validate_call(config={"arbitrary_types_allowed": True})
def transition_matrices(traj_dataset: Union[np.ndarray, Sequence[np.ndarray]], grid_size: int, bbox: Dict[str, float],
                        latgrids: Optional[np.ndarray] = None, longrids: Optional[np.ndarray] = None,
                        lon_first: bool = True) -> np.ndarray:
    """
    Compute the (start point -> end transition matrix) of a trajectory dataset, which is required for the computation
    of the Trip Error metric.

    :param traj_dataset: A dataset of trajectories with shape (n_trajectories, n_samples, 2).
    :param grid_size: Size of the grid to compute the heatmaps.
    :param bbox: Bounding box of the area of interest. Represented as a dictionary with keys 'lat_min', 'lat_max',
                 'lon_min', 'lon_max'. If you use x and y coordinates, use lon for the x-axis and lat for the y-axis.
    :param latgrids: Latitude grid boundaries. Including outer boundaries.
    :param longrids: Longitude grid boundaries. Including outer boundaries.
    :param lon_first: If True, trajectory points are in the format (longitude, latitude) else (latitude, longitude).
    :return: A transition matrix of shape (n_cells, n_cells), where n_cells = (len(latgrids) - 1) * (len(longrids) - 1)
    """
    # Create Grid if not provided
    if latgrids is None or longrids is None:
        latgrids, longrids = divide_grids(bbox, grid_size)
    latgrids = cast(np.ndarray, latgrids)
    longrids = cast(np.ndarray, longrids)
    assert grid_size is None or (len(latgrids) == grid_size + 1 and len(longrids) == grid_size + 1), \
        "Grid size should match the provided grids."
    assert bbox is None or (latgrids[0] == bbox['lat_min'] and latgrids[-1] == bbox['lat_max']), \
        "Latitude grids should match the provided boundaries."
    assert bbox is None or (longrids[0] == bbox['lon_min'] and longrids[-1] == bbox['lon_max']), \
        "Longitude grids should match the provided boundaries."

    # Number of cells in the grid
    n_cells = (len(latgrids) - 1) * (len(longrids) - 1)

    # Initialize the transition matrix
    transition_matrix = np.zeros((n_cells, n_cells))

    # If the trajectory points are in the format (latitude, longitude), reverse the order to (longitude, latitude)
    if not lon_first:
        if isinstance(traj_dataset, np.ndarray):
            traj_dataset = traj_dataset[:, :, ::-1]
        else:
            traj_dataset = [traj[:, ::-1] for traj in traj_dataset]

    # Iterate over each trajectory
    for traj in traj_dataset:
        start_point = traj[0]
        end_point = traj[-1]

        # Find the grid cells for the start and end points
        # Don't be strict about the points being within the boundary b/c of the Diffusion model's generated data
        # While for GAN/VAE you can simply use a tanh activation to ensure the points are within the boundary,
        # This is slightly more complicated for the Diffusion model, so we simply ignore the check for now.
        # If the model is trained properly, the points should be within the boundary anyway.
        start_cell = compute_grid_cell(lon=start_point[0], lat=start_point[1], latgrids=latgrids, longrids=longrids,
                                       check=False)
        end_cell = compute_grid_cell(lon=end_point[0], lat=end_point[1], latgrids=latgrids, longrids=longrids,
                                     check=False)

        # If both start and end points are within bounds, update the transition matrix
        transition_matrix[start_cell, end_cell] += 1

    # Normalize the transition matrix
    transition_matrix /= len(traj_dataset)

    return transition_matrix


@validate_call(config={"arbitrary_types_allowed": True})
def trip_error(real_traj_dataset: Union[np.ndarray, Sequence[np.ndarray]],
               gen_traj_dataset: Union[np.ndarray, Sequence[np.ndarray]], grid_size: int,
               bbox: Dict[str, float], latgrids: Optional[np.ndarray] = None, longrids: Optional[np.ndarray] = None,
               lon_first: bool = True, jsd_mode: JSDModes = JSD_MODE_STATISTICS) -> float:
    """
    Compute the Trip Error between real and generated trajectory datasets.

    :param real_traj_dataset: A dataset of real trajectories with shape (n_trajectories, n_samples, 2).
    :param gen_traj_dataset: A dataset of generated trajectories with shape (n_trajectories, n_samples, 2).
    :param grid_size: Size of the grid to compute the heatmaps.
    :param bbox: Bounding box of the area of interest. Represented as a dictionary with keys 'lat_min', 'lat_max',
                 'lon_min', 'lon_max'. If you use x and y coordinates, use lon for the x-axis and lat for the y-axis.
    :param latgrids: Latitude grid boundaries. Including outer boundaries.
    :param longrids: Longitude grid boundaries. Including outer boundaries.
    :param lon_first: If True, trajectory points are in the format (longitude, latitude) else (latitude, longitude).
    :param jsd_mode: The mode to use for the Jensen-Shannon Divergence (JSD) computation.
    :return: The Trip Error between the real and generated datasets.
    """
    # Compute the transition matrices for the real and generated datasets
    real_transition_matrix = transition_matrices(traj_dataset=real_traj_dataset, grid_size=grid_size,
                                                 bbox=bbox, latgrids=latgrids, longrids=longrids, lon_first=lon_first)
    gen_transition_matrix = transition_matrices(traj_dataset=gen_traj_dataset, grid_size=grid_size,
                                                bbox=bbox, latgrids=latgrids, longrids=longrids, lon_first=lon_first)

    # Flatten the transition matrices
    real_flat = real_transition_matrix.flatten()
    gen_flat = gen_transition_matrix.flatten()

    # Compute the Jensen-Shannon Distance/Divergence (JSD) between the real and generated transition matrices
    trip_error_value = JSD(real_flat, gen_flat, mode=jsd_mode)

    return trip_error_value
