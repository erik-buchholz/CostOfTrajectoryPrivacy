#!/usr/bin/env python3
"""Utility Metrics for the comparison of point distributions."""
from typing import Optional, Dict, cast

import numpy as np
import ot
from scipy.spatial.distance import jensenshannon

from privtrajgen.config import JSD_MODE_DISTRIBUTION, JSDModes, RANDOM_SEED
from privtrajgen.utils.grid import divide_grids
from .utils import verify_batch_sizes


def _JS_divergence(distribution1: np.ndarray, distribution2: np.ndarray) -> float:
    """Jensen Shannon Divergence between two distributions.
    Note that the input distributions are flattened to 1D arrays before computing the divergence.
    I.e., spatial information is lost in the process. We recommend using the SWD for spatially-aware comparisons.
    If the JSD is used, it requires computation of a heatmap from the points.
    See `JS_from_points` for more details.
    This JS Divergence is used by DiffTraj for the comparison of point distributions [1].

    [1] https://github.com/Yasoz/DiffTraj/issues/15#issuecomment-2249691574

    :param distribution1: Probability distribution 1 represented as an array of probabilities
        For a discrete distribution, the array contains the probabilities of each outcome.
        For a continuous distribution, the array contains the probabilities of each bin. Note that the choice
        of binning can affect the divergence.
    :param distribution2: Probability distribution 2 represented as an array of probabilities
    """
    return _JS_distance(distribution1, distribution2) ** 2


def _JS_distance(distribution1: np.ndarray, distribution2: np.ndarray) -> float:
    """Jensen Shannon Distance between two distributions.
    If the distributions are not 1D, they are flattened to 1D arrays before computing the distance.
    Use `JS_from_points` for spatial data!

    :param distribution1: Probability distribution 1 represented as an array of probabilities
    :param distribution2: Probability distribution 2 represented as an array of probabilities
    """
    return cast(float, jensenshannon(distribution1.flatten(), distribution2.flatten()))


def JSD(distribution1: np.ndarray, distribution2: np.ndarray, mode: JSDModes = JSDModes.DISTANCE) -> float:
    """Compute the Jensen Shannon Distance/Divergence between two distributions.
    The mode determines whether the distance or divergence is computed.
    Use 'divergence' as default for DiffTraj compatibility.

    :param distribution1: Probability distribution 1 represented as an array of probabilities
    :param distribution2: Probability distribution 2 represented as an array of probabilities
    :param mode: Mode of the JS computation. Either 'distance' or 'divergence'.
    :return: Jensen Shannon Distance/Divergence between the two distributions.
    """
    if mode == JSDModes.DISTANCE:
        return _JS_distance(distribution1, distribution2)
    elif mode == JSDModes.DIVERGENCE:
        return _JS_divergence(distribution1, distribution2)
    else:
        raise ValueError(f"Mode should be either 'distance' or 'divergence'. Got: {mode}.")


def compute_heatmap(points: np.ndarray, latgrids: np.ndarray, longrids: np.ndarray,
                    lon_first: bool = True) -> np.ndarray:
    """
    Compute a heatmap from a set of points using the provided latitude and longitude grids.

    :param points: Array of points with shape (n_samples, 2), where each point is (longitude, latitude).
    :param latgrids: Latitude grid boundaries. Including outer boundaries.
    :param longrids: Longitude grid boundaries. Including outer boundaries.
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :return: 2D heatmap array with density values.
    """
    if not lon_first:
        points = points[:, ::-1]
    heatmap, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[longrids, latgrids], density=True)
    return heatmap


def JS_from_points(points1: np.ndarray, points2: np.ndarray, grid_size: int, bbox: Dict[str, float],
                   latgrids: Optional[np.ndarray] = None, longrids: Optional[np.ndarray] = None,
                   mode: JSDModes = JSD_MODE_DISTRIBUTION,
                   lon_first: bool = True) -> float:
    """
    Compute the Jensen Shannon Distance/Divergence between two sets of points, i.e., samples drawn from
    underlying distributions.
    As the JSD compares probability distributions, we first compute the heatmaps of the points on a grid.
    The heatmaps are then used to compute the JSD.

    :param points1: Points from distribution 1 of shape (n_samples, 2)
    :param points2: Points from distribution 2 of shape (n_samples, 2)
    :param grid_size: Size of the grid to compute the heatmaps. DiffTraj uses $16 \\times 16$
    :param bbox: Bounding box of the area of interest. Represented as a dictionary with keys 'lat_min', 'lat_max',
                 'lon_min', 'lon_max'. If you use x and y coordinates, use lon for the x-axis and lat for the y-axis.
    :param latgrids: Latitude grid boundaries. Including outer boundaries.
    :param longrids: Longitude grid boundaries. Including outer boundaries.
    :param lon_first: If True, the trajectory is in the format (longitude, latitude), otherwise (latitude, longitude)
    :param mode: Mode of the JS computation. Either 'distance' or 'divergence'.
        Use 'divergence' as default for DiffTraj compatibility.
    :return: Jensen Shannon Distance/Divergence between the two sets of points.
    """
    # Check batch sizes
    verify_batch_sizes(points1, points2)
    # Create Grid
    if latgrids is None or longrids is None:
        latgrids, longrids = divide_grids(bbox, grid_size)
    # We now that both grids are np.ndarrays now
    latgrids = cast(np.ndarray, latgrids)
    longrids = cast(np.ndarray, longrids)
    assert grid_size is None or (len(latgrids) == grid_size + 1 and len(longrids) == grid_size + 1), \
        "Grid size should match the provided grids."
    assert bbox is None or (latgrids[0] == bbox['lat_min'] and latgrids[-1] == bbox['lat_max']), \
        "Latitude grids should match the provided boundaries."
    assert bbox is None or (longrids[0] == bbox['lon_min'] and longrids[-1] == bbox['lon_max']), \
        "Longitude grids should match the provided boundaries."
    # Compute Heatmaps
    heatmap1 = compute_heatmap(points1, latgrids, longrids, lon_first=lon_first)
    heatmap2 = compute_heatmap(points2, latgrids, longrids, lon_first=lon_first)
    # Compute JS Distance/Divergence
    return JSD(heatmap1, heatmap2, mode=mode)


def sliced_wasserstein_distance(samples1: np.ndarray, samples2: np.ndarray, seed: Optional[int] = RANDOM_SEED) -> float:
    """Computes Monte Carlo approximation of Sliced Wasserstein Distance (SWD) between samples drawn from
        two nD distribution.

    :param samples1: Samples from distribution 1 of shape (n_samples, n_dim)
    :param samples2: Samples from distribution 2 of shape (n_samples, n_dim)
    :param seed: Random seed for reproducibility
    :return: Monte-Carlo approximation of Sliced Wasserstein Distance (number of projections = 50)
    """
    # Test that both shapes have two dimensions
    assert samples1.ndim == 2 and samples2.ndim == 2, "Samples should have two dimensions."
    verify_batch_sizes(samples1, samples2)

    N_PROJECTIONS = 50
    swd = cast(float, ot.sliced_wasserstein_distance(samples1, samples2, n_projections=N_PROJECTIONS, seed=seed))
    return swd
