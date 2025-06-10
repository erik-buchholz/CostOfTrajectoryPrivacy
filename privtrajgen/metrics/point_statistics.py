#!/usr/bin/env python3
"""Point-level Statistics Utility Metrics."""
import logging
from typing import List, Union, Dict, Callable, Sequence, Any

import numpy as np
from pydantic import validate_call
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

from privtrajgen.metrics import PointDistanceMetric, MAE, MRE, DATASET_1, DATASET_2, _haversine
from privtrajgen.utils.grid import divide_grids, compute_grid_id

log = logging.getLogger(__name__)


@validate_call(config={"arbitrary_types_allowed": True})
def get_random_query_points(bbox: Dict[str, float], n_points: int, lon_first: bool = True) -> List[np.ndarray]:
    """
    Generate random query points within the bounding box.

    :param bbox: The bounding box of the area of interest, with keys 'lon_min', 'lon_max', 'lat_min', 'lat_max'.
    :param n_points: The number of random query points to generate.
    :param lon_first: If True, dataset points are in the format (longitude, latitude). If False, (latitude, longitude).
    :return: A list of random query points, each with shape (2,) representing a point of interest (lon, lat).
    """
    lon_min, lon_max = bbox['lon_min'], bbox['lon_max']
    lat_min, lat_max = bbox['lat_min'], bbox['lat_max']

    query_points = []
    for _ in range(n_points):
        lon = np.random.uniform(lon_min, lon_max)
        lat = np.random.uniform(lat_min, lat_max)
        query_point = np.array([lon, lat]) if lon_first else np.array([lat, lon])
        query_points.append(query_point)

    return query_points


@validate_call(config={"arbitrary_types_allowed": True})
def range_query(point_dataset: np.ndarray, query_points: Union[np.ndarray, List[np.ndarray]],
                radii: Union[float, Sequence[float | int]],
                metric: Union[PointDistanceMetric, Callable] = PointDistanceMetric.EUCLIDEAN,
                lon_first: bool = True, **metric_kwargs: Any) -> List[List[Sequence[int]]]:
    """
    Perform a range query to find points in the dataset within a certain radius (or multiple radii) of the query point(s).

    :param point_dataset: A dataset of points with shape (n_points, 2) representing (longitude, latitude) or (latitude, longitude).
    :param query_points: A query point with shape (2,) or a list of query points, each representing a point of interest (lon, lat) or (lat, lon).
    :param radii: A single radius or a list of radii within which points are considered, in the same units as the distance metric.
    :param metric: The distance metric to use (e.g., Euclidean or Haversine). Defaults to Euclidean distance.
    :param lon_first: If True, dataset points are in the format (longitude, latitude). If False, (latitude, longitude).
    :param metric_kwargs: Additional keyword arguments for the distance function.
    :return: List[List[int]]: results[i][j] is a list of indices for the points around query_points[i] within radii[j].
    """
    # Ensure query_points and radii are lists
    if not isinstance(query_points, list):
        query_points = [query_points]
    if not isinstance(radii, Sequence):
        radii = [radii]

    # All metrics expect (lon, lat) format, so reverse the dataset if necessary
    if not lon_first:
        point_dataset = point_dataset[:, ::-1]
        query_points = [qp[::-1] for qp in query_points]

    # Set the haversine metrics
    if metric == PointDistanceMetric.HAVERSINE:
        metric = _haversine

    results: List[List[Sequence[int]]] = []

    # Compute the range query for each combination of query point and radius
    for query_point in query_points:
        # Compute distances between the query point and all points in the dataset
        distances = cdist(point_dataset, np.array([query_point]), metric=metric, **metric_kwargs).flatten()

        query_results: List[Sequence[int]] = []
        for radius in radii:
            # Find indices of points within the specified radius
            result_indices = np.where(distances <= radius)[0]
            query_results.append(result_indices.tolist())

        results.append(query_results)

    return results


@validate_call(config={"arbitrary_types_allowed": True})
def range_query_error(point_dataset1: np.ndarray, point_dataset2: np.ndarray,
                      query_points: Union[np.ndarray, List[np.ndarray]],
                      radii: Union[float, Sequence[float], Sequence[int]],
                      metric: Union[PointDistanceMetric, Callable] = PointDistanceMetric.EUCLIDEAN,
                      lon_first: bool = True,
                      b_factor: float = 0.001, **metric_kwargs: Any) -> dict:
    """
    Compute Mean Absolute Error (MAE) and Mean Relative Error (MRE) between two datasets based on range queries.

    :param point_dataset1: The first dataset of points (e.g., real data) with shape (n_points, 2).
    :param point_dataset2: The second dataset of points (e.g., synthetic/sanitised data) with shape (n_points, 2).
    :param query_points: A query point with shape (2,) or a list of query points, each representing a point of interest (lon, lat).
    :param radii: A single radius or a list of radii within which points are considered.
    :param metric: The distance metric to use (e.g., Euclidean or Haversine). Defaults to Euclidean distance.
    :param lon_first: If True, dataset points are in the format (longitude, latitude). If False, (latitude, longitude).
    :param b_factor: A sanity bound factor used to mitigate the effects of extremely small query results in relative error.
        0.1% of dataset as a default value.
    :param metric_kwargs: Additional keyword arguments for the distance function.
    :return: A dictionary with 'MAE' and 'MRE' as keys, holding the respective error values.
    """

    # Perform the range queries on both datasets
    results1 = range_query(point_dataset1, query_points, radii, metric, lon_first, **metric_kwargs)
    results2 = range_query(point_dataset2, query_points, radii, metric, lon_first, **metric_kwargs)

    if not isinstance(radii, Sequence):
        radii = [radii]

    num_queries = len(query_points)
    num_radii = len(radii)

    absolute_errors = []
    relative_errors = []

    # Compute the sanity bound 'b' as a fraction of the dataset size
    b = int(len(point_dataset1) * b_factor)

    # Iterate over each query point and radius
    empty_results = 0
    for i in range(num_queries):
        for j in range(num_radii):
            # Get the number of points found in the query for both datasets
            q_real = len(results1[i][j])
            q_syn = len(results2[i][j])

            if q_real == 0:
                empty_results += 1

            # Compute absolute and relative errors
            absolute_error = abs(q_real - q_syn)
            relative_error = absolute_error / max(q_real, b)  # Using the sanity bound 'b'

            # Append the errors to the lists
            absolute_errors.append(absolute_error)
            relative_errors.append(relative_error)

    # Compute Mean Absolute Error (MAE)
    mae = np.mean(absolute_errors) if absolute_errors else 0

    # Compute Mean Relative Error (MRE)
    mre = np.mean(relative_errors) if relative_errors else 0

    # Log a warning if there are too many empty results
    if empty_results > 0.2 * num_queries * num_radii:
        log.info(f"[Range Query Error] {empty_results} out of {num_queries * num_radii} results are empty.")

    # Store detailed results as two dicts:
    # Dataset1[str(query_points[i])][str(radii[j])] = len(result[i][j])
    dataset1 = {}
    dataset2 = {}
    for i in range(num_queries):
        dataset1[str(query_points[i])] = {str(radii[j]): len(results1[i][j]) for j in range(num_radii)}
        dataset2[str(query_points[i])] = {str(radii[j]): len(results2[i][j]) for j in range(num_radii)}

    # Return the results as a dictionary
    return {
        MAE: mae,
        MRE: mre,
        DATASET_1: dataset1,
        DATASET_2: dataset2
    }


@validate_call(config={"arbitrary_types_allowed": True})
def safe_kde(point_dataset: np.ndarray, **kde_kwargs: Any) -> gaussian_kde:
    """
    Create a KDE object from a dataset of points, applying a small jitter if a singular matrix error is encountered.

    :param point_dataset: A dataset of points with shape (n_features, n_samples).
    :param kde_kwargs: Additional keyword arguments for the KDE function.
    :return: A KDE object created from the dataset.
    """
    try:
        kde = gaussian_kde(point_dataset, **kde_kwargs)
    except np.linalg.LinAlgError:
        # Apply a small jitter
        log.warning("[Hotspot Analysis] Singular matrix encountered. Applying jitter to the dataset.")
        point_dataset += np.random.normal(0, 1e-6, point_dataset.shape)
        kde = gaussian_kde(point_dataset, **kde_kwargs)
    return kde


@validate_call(config={"arbitrary_types_allowed": True})
def hotspot_analysis(point_dataset: np.ndarray,
                     grid_sizes: Union[int, List[int]],
                     bbox: Dict[str, float],
                     percentile: float = 0.95, lon_first: bool = True,
                     kde_params: Dict = {'bw_method': 'scott'}
                     ) -> List[List[int]]:
    """
    Perform hotspot analysis on a dataset of points.

    :param point_dataset: A dataset of points with shape (n_points, 2) representing (longitude, latitude) or (latitude, longitude).
    :param grid_sizes: A list of grid sizes to use for the hotspot analysis.
    :param bbox: The bounding box of the area of interest, with keys 'lon_min', 'lon_max', 'lat_min', 'lat_max'.
    :param percentile: The percentile value to use as the threshold for hotspot detection.
    :param lon_first: If True, dataset points are in the format (longitude, latitude). If False, (latitude, longitude).
    :param kde_params: Additional keyword arguments for the KDE function. Mainly used for testing.
    :return List[List[int]]: results[i] is a list of grid cell IDs (as integers) corresponding to hotspots for grid_sizes[i].
    """
    # Ensure grid_sizes is a list
    if not isinstance(grid_sizes, list):
        grid_sizes = [grid_sizes]

    # Transpose data to (lon, lat) format if necessary
    if not lon_first:
        point_dataset = point_dataset[:, ::-1]

    results = []

    # Perform the hotspot analysis for each grid size
    for grid_size in grid_sizes:
        latgrids, longrids = divide_grids(bbox, grid_size)

        # Determine the grid's center points
        lon_centers = (longrids[:-1] + longrids[1:]) / 2
        lat_centers = (latgrids[:-1] + latgrids[1:]) / 2

        # Create a meshgrid of the grid center points
        lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
        # Output: lon_mesh = [[lon_0, lon_1, lon_2, ...], [lon_0, lon_1, lon_2, ...], ...]

        # Perform KDE (Kernel Density Estimation) to estimate the density of points in the dataset
        # scipy.stats.gaussian_kde requires the input data to be in the format (n_features, n_samples)
        # Default bandwidth method is 'scott'. Fix for future reproducibility.
        kde = safe_kde(point_dataset.T, **kde_params)

        # Evaluate KDE over the grid points
        grid_points = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()]).T
        # After this, grid points will contain all the grid center points in the format (lon, lat) ordered in the
        # same way as we enumerate the grid cells in compute_grid_id

        # Evaluate the KDE at the grid points
        # Again, transpose the grid points to (n_features, n_samples) format
        densities = kde(grid_points.T).reshape(grid_size, grid_size)

        # Determine the density threshold (percentile)
        density_threshold = np.percentile(densities, percentile * 100)

        # Find the grid cells that are hotspots (density > threshold)
        hotspots = np.argwhere(densities > density_threshold)
        # Output: hotspots = [[lat_idx_0, lon_idx_0], [lat_idx_1, lon_idx_1], ...]
        # Latitude index is the row index and longitude index is the column index

        # Convert the grid cell indices to grid IDs
        hotspot_ids = [compute_grid_id(lon_idx=lon_idx, lat_idx=lat_idx, n_lon_cols=grid_size)
                       for lat_idx, lon_idx in hotspots]

        if len(hotspot_ids) == 0:
            log.warning(f"No hotspots found for grid size {grid_size}.")

        results.append(hotspot_ids)

    return results


@validate_call(config={"arbitrary_types_allowed": True})
def sorensen_dice_coefficient(H_real: set, H_syn: set) -> float:
    """
    Compute the Sørensen-Dice coefficient between two sets of hotspots.

    :param H_real: A set of indices
    :param H_syn: A set of indices
    :return: The Sørensen-Dice coefficient as a float.
    """
    intersection_size = len(H_real & H_syn)
    return 2 * intersection_size / (len(H_real) + len(H_syn))


@validate_call(config={"arbitrary_types_allowed": True})
def hotspot_analysis_error(
        point_dataset1: np.ndarray,
        point_dataset2: np.ndarray,
        grid_sizes: Union[int, List[int]],
        bbox: Dict[str, float],
        percentile: float = 0.95,
        lon_first: bool = True,
) -> Dict[int, float]:
    """
    Compare the real and synthetic/sanitised datasets using the Sørensen-Dice coefficient for hotspot analysis.

    :param point_dataset1: The first dataset of points (e.g., real data) with shape (n_points, 2),
                           representing (longitude, latitude) or (latitude, longitude).
    :param point_dataset2: The second dataset of points (e.g., synthetic/sanitised data) with shape (n_points, 2),
                           representing (longitude, latitude) or (latitude, longitude).
    :param grid_sizes: A list of grid sizes (granularities) to use for the hotspot analysis.
    :param bbox: The bounding box of the area of interest, with keys 'lon_min', 'lon_max', 'lat_min', 'lat_max'.
    :param percentile: The percentile value to use as the threshold for hotspot detection.
    :param lon_first: If True, dataset points are in the format (longitude, latitude). If False, (latitude, longitude).
    :return: A dictionary where keys are grid sizes and values are Sørensen-Dice coefficients for the corresponding grid size.
    """
    # Ensure grid_sizes is a list
    if not isinstance(grid_sizes, list):
        grid_sizes = [grid_sizes]

    sorensen_dice_coefficients = {}

    # Perform hotspot analysis for both datasets
    for grid_size in grid_sizes:
        # Get hotspots for both datasets
        H_real = hotspot_analysis(point_dataset1, grid_size, bbox, percentile, lon_first)[0]
        H_syn = hotspot_analysis(point_dataset2, grid_size, bbox, percentile, lon_first)[0]

        # To be safe that there are no duplicates, convert the hotspots to sets
        H_real = set(H_real)
        H_syn = set(H_syn)

        # Compute the Sørensen-Dice coefficient using the separate method
        dice_coefficient = sorensen_dice_coefficient(H_real, H_syn)

        # Store the result for this grid size
        sorensen_dice_coefficients[grid_size] = dice_coefficient

    return sorensen_dice_coefficients
