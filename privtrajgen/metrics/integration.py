#!/usr/bin/env python3
"""Integration of all utility metrics."""
import json
import logging
import os
from dataclasses import asdict
from timeit import default_timer as timer
from typing import List, Union, Optional, Sequence, Dict, Any

import numpy as np
from deepdiff import DeepDiff
from pydantic import validate_call

# Import evaluation configuration
from privtrajgen.utils import time_to_str
# Point distance metrics
## Point-Level Utility Metrics (Preservation)
from .distribution import sliced_wasserstein_distance, JS_from_points
# Keys
from .keys import *
## Point-Level Utility Metrics (Statistics)
from .point_statistics import hotspot_analysis_error, range_query_error, get_random_query_points
## Trajectory-Level Preservation Utility Metrics
from .trajectory_preservation import (hausdorff_distance, batched_trajectory_distance_minimized,
                                      batched_trajectory_distance, reorder_dataset_to_matching, aggregate_metric)
## Trajectory-Level Statistics Utility Metrics
from .trajectory_statistics import trip_error, trajectory_statistics_error
from .utils import verify_batch_sizes
from ..config import NORMALISE_TRAJECTORY_DISTANCES, NUM_RANGE_QUERIES, JSD_GRID_SIZES, RANGE_QUERY_RADII, \
    SANITY_BOUND_FACTOR, HOTSPOT_GRID_SIZES, HOTSPOT_PERCENTILE, TRIP_ERROR_GRID_SIZE, N_BINS_TRAJ_STATS, \
    HD_POINT_LIMIT, POINT_LIMIT, TRAJECTORY_LIMIT, JSD_MODE_DISTRIBUTION, JSD_MODE_STATISTICS, config
from ..evaluation.config import BaseConfig, PrivTraceConfig
from ..models.naming import generate_model_name

log = logging.getLogger(__name__)


def logc(header:str, *args: Any, column_width: int = 60, level: int =logging.INFO) -> None:
    txt = f"{header:<{column_width}}" + "".join([str(arg) for arg in args])
    log.log(level, txt)


@validate_call(config={"arbitrary_types_allowed": True})
def compute_all_metrics(
        trajectory_dataset1: Union[np.ndarray, List[np.ndarray]],
        trajectory_dataset2: Union[np.ndarray, List[np.ndarray]],
        lon_first: bool,
        geographical_coordinates: bool,
        are_matched: bool,
        # Metric Parameters
        bbox: dict,
        random_seed: Optional[int] = None,
        normalise_trajectory_distances: bool = NORMALISE_TRAJECTORY_DISTANCES,
        js_grid_sizes: Union[Sequence[int], int] = JSD_GRID_SIZES,
        n_range_queries: int = NUM_RANGE_QUERIES,
        range_query_radii: Sequence[Union[float, int]] = RANGE_QUERY_RADII,
        sanity_bound_factor: float = SANITY_BOUND_FACTOR,
        hotspot_grid_sizes: List[int] = HOTSPOT_GRID_SIZES,
        hotspot_percentile: float = HOTSPOT_PERCENTILE,
        trip_error_grid_size: int = TRIP_ERROR_GRID_SIZE,
        num_bins_traj_stats: int = N_BINS_TRAJ_STATS,
        metric_kwargs: dict = dict(),
        hd_point_limit: Optional[int] = HD_POINT_LIMIT,
        point_limit: Optional[int] = POINT_LIMIT,
        trajectory_limit:Optional[int] = TRAJECTORY_LIMIT
) -> dict:
    """
    Compute all utility metrics for two trajectory datasets.
    Note that the dataset shapes MUST be (n_trajectories, length, 2).
    If you approach uses a different shape, please reshape the dataset accordingly.

    :param trajectory_dataset1: The first trajectory dataset with shape (n_trajectories, length, 2) or a list of
                                trajectories with shape (length, 2), where length might vary between trajectories.
    :param trajectory_dataset2: The second trajectory dataset with shape (n_trajectories, length, 2) or a list of
                                trajectories with shape (length, 2), where length might vary between trajectories.
    :param lon_first: If True, last dimension of the trajectory dataset is (longitude, latitude) else
                      (latitude, longitude).
    :param geographical_coordinates: If True, the coordinates are geographical coordinates (lat, lon).
                                    --> haversine distance is used for distance computation.
    :param are_matched: If True, the trajectories are already matched. If False, the trajectories are matched based on
                        the Euclidean distance or Haversine distance.
    :param random_seed: Random seed for reproducibility.
    :param bbox: The bounding box of the area of interest, with keys 'lon_min', 'lon_max', 'lat_min', 'lat_max'.
    :param normalise_trajectory_distances: If True, normalize trajectory distances by the trajectory length.
    :param js_grid_sizes: Grid sizes for JSD computation.
    :param n_range_queries: Number of range queries or a list of specific query points.
    :param range_query_radii: List of radii for range queries.
    :param sanity_bound_factor: Factor for sanity bound for the computation of relative error in range queries.
    :param hotspot_grid_sizes: List of grid sizes for hotspot analysis.
    :param hotspot_percentile: Percentile value for hotspot detection.
    :param trip_error_grid_size: Grid size for trip error computation.
    :param num_bins_traj_stats: Number of bins for the histogram in the trajectory statistics.
    :param metric_kwargs: Additional keyword arguments for metric computations.
    :param hd_point_limit: Limit for the number of points for the Hausdorff distance computation due to computational
                            complexity. If the number of points exceeds this limit, the points are randomly sampled.
    :param point_limit: Limit the number of points in the dataset. None for no limit. Due to overhead.
    :param trajectory_limit: Limit the number of trajectories in the dataset. None for no limit. Due to overhead.
    :return: A dictionary containing all computed metrics: result['metric_name'] = metric_value
    """
    # Verify the batch sizes
    verify_batch_sizes(trajectory_dataset1, trajectory_dataset2)
    assert trajectory_dataset1[0].shape[1] == 2, "Trajectory dataset must have shape (n_trajectories, length, 2)."
    assert trajectory_dataset2[0].shape[1] == 2, "Trajectory dataset must have shape (n_trajectories, length, 2)."

    # Limit number of trajectories and points if necessary
    if trajectory_limit is not None and trajectory_limit < len(trajectory_dataset1):
        # Random sampling - keep order!
        idx = np.random.choice(len(trajectory_dataset1), trajectory_limit, replace=False)
        trajectory_dataset1 = [trajectory_dataset1[i] for i in idx]
        trajectory_dataset2 = [trajectory_dataset2[i] for i in idx]
        log.warning(f"Number of trajectories exceeds the limit of {trajectory_limit}. Randomly sampling trajectories.")

    if geographical_coordinates:
        point_dist_metric = PointDistanceMetric.HAVERSINE
        traj_dist_metric = TrajectoryDistanceMetric.HAVERSINE
    else:
        point_dist_metric = PointDistanceMetric.EUCLIDEAN
        traj_dist_metric = TrajectoryDistanceMetric.EUCLIDEAN

    results: Dict[Any, Any] = {}

    # Point-Level Utility Metrics (Preservation)
    ####################################################################################################

    log.debug("Generate point datasets.")
    # Create point datasets. Shape: (n_trajectories * length, 2). Note that length might vary between trajectories.
    if isinstance(trajectory_dataset1, Sequence):
        points_dataset1 = np.vstack(trajectory_dataset1)
    else:
        assert trajectory_dataset1.shape[-1] == 2, "Trajectory dataset must have shape (n_trajectories, length, 2)."
        points_dataset1 = trajectory_dataset1.reshape(-1, 2)
    if isinstance(trajectory_dataset2, Sequence):
        points_dataset2 = np.vstack(trajectory_dataset2)
    else:
        assert trajectory_dataset2.shape[-1] == 2, "Trajectory dataset must have shape (n_trajectories, length, 2)."
        points_dataset2 = trajectory_dataset2.reshape(-1, 2)

    # Ensure that both point datasets have the same length
    if point_limit is not None and point_limit < min(points_dataset1.shape[0], points_dataset2.shape[0]):
        log.warning(f"Number of points exceeds the limit of {point_limit}. Randomly sampling points.")
        idx = np.random.choice(min(points_dataset1.shape[0], points_dataset2.shape[0]), point_limit, replace=False)
        points_dataset1 = points_dataset1[idx]
        points_dataset2 = points_dataset2[idx]
    elif points_dataset1.shape[0] != points_dataset2.shape[0]:
        log.warning(f"Point datasets have different lengths: {points_dataset1.shape[0]} vs {points_dataset2.shape[0]}."
                    " Truncating the longer dataset.")
        min_l = min(points_dataset1.shape[0], points_dataset2.shape[0])
        # Shuffle (point order doesn't matter for any of the point metrics) and truncate
        np.random.shuffle(points_dataset1)
        np.random.shuffle(points_dataset2)
        points_dataset1 = points_dataset1[:min_l]
        points_dataset2 = points_dataset2[:min_l]

    # Jensen-Shannon Divergence
    metric_start = timer()
    if not isinstance(js_grid_sizes, Sequence):
        js_grid_sizes = [js_grid_sizes]
    results[JSD_DISTRIBUTION] = {}
    for grid_size in js_grid_sizes:
        results[JSD_DISTRIBUTION][grid_size] = JS_from_points(
            points_dataset1, points_dataset2, lon_first=lon_first, mode=JSD_MODE_DISTRIBUTION, grid_size=grid_size,
            bbox=bbox
        )
    logc(f"Computed JSD of Heatmaps. Time elapsed:", f"{time_to_str(timer() - metric_start)}", level=logging.DEBUG)

    # Sliced Wasserstein Distance
    metric_start = timer()
    results[SWD_DISTRIBUTION] = sliced_wasserstein_distance(points_dataset1, points_dataset2, seed=random_seed)
    logc(f"Computed Sliced Wasserstein Distance. Time elapsed:", f"{time_to_str(timer() - metric_start)}",
         level=logging.DEBUG)

    # Point Hausdorff Distance
    if hd_point_limit is not None and points_dataset1.shape[0] > hd_point_limit:
        log.info(f"Number of points exceeds the limit of {hd_point_limit:,} for HD computation. "
                    f"Randomly sampling points.")
        idx = np.random.choice(points_dataset1.shape[0], hd_point_limit, replace=False)
        points_dataset1_sampled = points_dataset1[idx]
        idx = np.random.choice(points_dataset2.shape[0], hd_point_limit, replace=False)
        points_dataset2_sampled = points_dataset2[idx]
    else:
        points_dataset1_sampled = points_dataset1
        points_dataset2_sampled = points_dataset2
    metric_start = timer()
    results[HD_POINTS] = hausdorff_distance(points_dataset1_sampled, points_dataset2_sampled, lon_first=lon_first,
                                            use_haversine=geographical_coordinates)
    logc(f"Computed Hausdorff Distance of Points. Time elapsed:", f"{time_to_str(timer() - metric_start)}",
         level=logging.DEBUG)

    # Generate random query points for Range Queries
    metric_start = timer()
    if isinstance(n_range_queries, int):
        random_points = get_random_query_points(bbox=bbox, n_points=n_range_queries, lon_first=lon_first)
    else:
        raise ValueError("n_range_queries must be an integer.")
    logc(f"Generated random query points for Range Queries. Time elapsed:", f"{time_to_str(timer() - metric_start)}",
         level=logging.DEBUG)

    # Range Query Error
    metric_start = timer()
    range_query_res = range_query_error(points_dataset1, points_dataset2, query_points=random_points,
                                        radii=range_query_radii, metric=point_dist_metric, lon_first=lon_first,
                                        b_factor=sanity_bound_factor, **metric_kwargs)
    logc(f"Computed Range Query Error. Time elapsed:", f"{time_to_str(timer() - metric_start)}", level=logging.DEBUG)
    results[RANGE_QUERY] = range_query_res  # Contains keys "MAE", "MRE"

    # Hotspot Analysis Error
    metric_start = timer()
    hotspot_analysis_results = hotspot_analysis_error(points_dataset1, points_dataset2, lon_first=lon_first,
                                                      bbox=bbox, grid_sizes=hotspot_grid_sizes,
                                                      percentile=hotspot_percentile)
    logc(f"Computed Hotspot Analysis. Time elapsed:", f"{time_to_str(timer() - metric_start)}", level=logging.DEBUG)
    # For hotspot analysis, we have multiple metrics, but GeoPointGAN does not average them either.
    results[HOTSPOT_ANALYSIS] = hotspot_analysis_results

    # Trajectory-Level Utility Metrics (Preservation)
    ####################################################################################################
    # 1. Create a matching if not provided
    if not are_matched:
        metric_start = timer()
        # Match based on the simplest metric: Euclidean distance or Haversine distance
        # Don't use the normalized versions, as we want to consider length differences during matching
        matching, minimized_distances, _ = batched_trajectory_distance_minimized(
            trajectory_dataset1, trajectory_dataset2, distance_metric=traj_dist_metric, lon_first=lon_first,
            normalise=False, **metric_kwargs
        )
        # This also gives us the first value we need --> It doesn't because of the normalization
        # results[ED_TRAJECTORIES] = minimized_distances
        # Reorder the second dataset based on the matching (matching is a list of indices i->j)
        trajectory_dataset2 = reorder_dataset_to_matching(trajectory_dataset2, matching)
        logc(f"Computed Trajectory Matching. Time elapsed:", f"{time_to_str(timer() - metric_start)}",
             level=logging.DEBUG)

    # Euclidean/Haversine Distance
    metric_start = timer()
    results[ED_TRAJECTORIES] = batched_trajectory_distance(trajectory_dataset1, trajectory_dataset2,
                                                           distance_metric=traj_dist_metric, lon_first=lon_first,
                                                           normalise=normalise_trajectory_distances, **metric_kwargs)
    logc(f"Computed Euclidean/Haversine Distance. Time elapsed:", f"{time_to_str(timer() - metric_start)}",
         level=logging.DEBUG)

    # Hausdorff Distance of Trajectories
    metric_start = timer()
    results[HD_TRAJECTORIES] = batched_trajectory_distance(
        trajectory_dataset1, trajectory_dataset2, distance_metric=TrajectoryDistanceMetric.HAUSDORFF,
        lon_first=lon_first,
        use_haversine=geographical_coordinates, **metric_kwargs
    )
    logc(f"Computed Hausdorff Distance of Trajectories. Time elapsed:", f"{time_to_str(timer() - metric_start)}",
         level=logging.DEBUG)

    # DTW Distance of Trajectories
    metric_start = timer()
    results[DTW_TRAJECTORIES] = batched_trajectory_distance(
        trajectory_dataset1, trajectory_dataset2, distance_metric=TrajectoryDistanceMetric.DTW, lon_first=lon_first,
        use_haversine=geographical_coordinates, **metric_kwargs
    )
    logc(f"Computed DTW Distance of Trajectories. Time elapsed:", f"{time_to_str(timer() - metric_start)}",
         level=logging.DEBUG)

    # Store the aggregated metrics for the trajectory distance metrics
    for metric_key, aggregation_key in TRAJ_METRIC_AGGREGATION_MAPPING.items():
        results[aggregation_key] = aggregate_metric(results[metric_key])

    # Trajectory-Level Statistics Utility Metrics
    ####################################################################################################

    # Trip Error
    metric_start = timer()
    results[TRIP_ERROR] = trip_error(trajectory_dataset1, trajectory_dataset2, grid_size=trip_error_grid_size,
                                     bbox=bbox, lon_first=lon_first)
    logc(f"Computed Trip Error. Time elapsed:", f"{time_to_str(timer() - metric_start)}", level=logging.DEBUG)

    # Trajectory Statistics Error
    metric_start = timer()
    statistics = trajectory_statistics_error(
        trajectory_dataset1, trajectory_dataset2, lon_first=lon_first, metric=point_dist_metric, **metric_kwargs,
        jsd_mode=JSD_MODE_STATISTICS, num_bins=num_bins_traj_stats,
    )
    results[TTD] = statistics[TTD]
    results[DIAMETER] = statistics[DIAMETER]
    logc(f"Computed Trajectory Statistics Error. Time elapsed:", f"{time_to_str(timer() - metric_start)}",
         level=logging.DEBUG)

    return results


@validate_call
def save_metrics(results: dict, filename: str, metadata: Optional[dict] = None,
                 output_dir: str = config.RESULT_DIR, per_traj_metrics: bool = True) -> None:
    """
    Save the results to a file.

    :param results: The `results` dictionary.
    :param filename: The filename. The extension will be added automatically.
    :param metadata: The metadata dictionary.
    :param output_dir: The path to the directory where the file should be saved.
    :param per_traj_metrics: If True, save the per-trajectory metrics in a separate file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Check if the filename has the correct extension
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = os.path.join(output_dir, filename)

    # Save the results, but exclude the details per-trajectory metrics = TRAJ_PRESERVATION_METRICS_EXACT
    # as they are already saved in a separate file
    results_agg = {k: v for k, v in results.items() if k not in TRAJ_PRESERVATION_METRICS_EXACT}

    # Convert all numpy floats to Python floats
    for k, v in results_agg.items():
        if isinstance(v, np.float32) or isinstance(v, np.float64):
            results_agg[k] = float(v)

    # Add metadata if available
    if metadata is not None:
        results_agg[METADATA] = metadata

    with open(filepath, "w") as f:
        json.dump(results_agg, f, indent=4)

    if per_traj_metrics:
        # Save the per-trajectory metrics in a separate file (.npz)
        traj_filename = filepath.replace(".json", f"{TRAJ_PRES_EXTENSION}")
        per_traj_results = {k: v for k, v in results.items() if k in TRAJ_PRESERVATION_METRICS_EXACT}
        np.savez(traj_filename, **per_traj_results)

    log.info(f"Results saved to '{filepath}'.")


@validate_call
def load_metrics(filename: str, directory: str = config.RESULT_DIR, per_traj_metrics: bool = True) -> dict:
    """
    Load the results from a file.

    :param filename: The filename. The extension will be added automatically.
    :param directory: The path to the directory where the file is saved.
    :param per_traj_metrics: If True, load the per-trajectory metrics from a separate file.
    :return: The `results` dictionary.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Check if the filename has the correct extension
    if not filename.endswith(".json"):
        filename += ".json"

    # Check if filename is a full path
    if not os.path.isabs(filename):
        filepath = os.path.join(directory, filename)
    else:
        filepath = filename

    with open(filepath, "r") as f:
        results = json.load(f)

    if per_traj_metrics:
        # Load the per-trajectory metrics from a separate file (.npz)
        traj_filename = filepath.replace(".json", f"{TRAJ_PRES_EXTENSION}")
        per_traj_results = np.load(traj_filename)
        for k, v in per_traj_results.items():
            results[k] = v

    return results


@validate_call(config={"arbitrary_types_allowed": True})
def load_case_metrics(case: BaseConfig | PrivTraceConfig, directory: str = config.RESULT_DIR, per_traj_metrics: bool = False,
                      eval_run: Optional[int] = 1) -> dict:
    """
    Load the results for a given evaluation case from a file.

    :param case: The evaluation case.
    :param directory: The path to the directory where the file is saved.
    :param per_traj_metrics: If True, load the per-trajectory metrics from a separate file.
    :param eval_run: The evaluation run.
    :return: The `results` dictionary.
    """
    # Get the filename
    filename = generate_model_name(case)
    if eval_run is not None and eval_run > 0:
        filename += f"_{eval_run:02d}"
    loaded_data = load_metrics(filename, directory, per_traj_metrics)

    # Verify that the metadata matches the case
    metadata = loaded_data[METADATA]
    expected_metadata = json.loads(json.dumps(asdict(case)))

    # Compare the metadata
    # Ignore seed, num_runs, and gpu
    excluded_paths = {"root['seed']", "root['num_runs']", "root['gpu']"}
    if not case.conditional or not case.cond.dp:  # type: ignore
        # For unconditional models or those without DP mechanism, we can ignore the location
        excluded_paths.add("root['cond']['location']")
    if not case.dp.dp:
        # For models without DP mechanism, we can ignore the DP-related parameters
        excluded_paths.add("root['dp']['adaptive_grad_norm']")
        excluded_paths.add("root['dp']['max_grad_norm']")
    diff = DeepDiff(metadata, expected_metadata, exclude_paths=excluded_paths)
    if diff:
        raise ValueError(f"Metadata does not match the case: {diff}. Case: {case.name}")

    return loaded_data
