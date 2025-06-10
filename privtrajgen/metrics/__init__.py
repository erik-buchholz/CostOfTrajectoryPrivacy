#!/usr/bin/env python3
"""Implementation of Utility Metrics.

Following recommendations from [1], we implement the following metrics:

1) Point-Level Utility Metrics
    1a) Preservation
        - Jensen-Shannon Divergence (b/c Used By DiffTraj [2]) [Smaller Is Better]
        - Sliced Wasserstein Distance [Smaller Is Better]
    1b) Statistics
        - Range Query Error (MAE/MRE) [Smaller Is Better]
        - Hotspot Analysis (Sorensen-Dice Coefficient) [Larger Is Better]

2) Trajectory-Level Utility Metrics
    2a) Preservation
        - Hausdorff Distance [Smaller Is Better]
        - (Normalised) Euclidean Distance [Smaller Is Better]
        - (Normalised) Haversine Distance [Smaller Is Better]
        - Dynamic Time Warping [Smaller Is Better]
    2b) Statistics
        - Total Travelled Distance [Smaller Is Better]
        - Trajectory Diameter [Smaller Is Better]
        - Trip Error [Smaller Is Better]

[1] E. Buchholz, A. Abuadbba, S. Wang, S. Nepal, and S. S. Kanhere,
“SoK: Can Trajectory Generation Combine Privacy and Utility?,”
PoPETS, vol. 2024, no. 3, pp. 75--93, Jul. 2024, doi: 10.56553/popets-2024-0068.
[2] Y. Zhu, Y. Ye, S. Zhang, X. Zhao, and J. J. Q. Yu,
“DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model,”
in Advances in Neural Information Processing Systems, in 1, vol. 23.
New Orleans, USA, 2023, p. 21. doi: 10.5555/3666122.3668965.
"""
# Import the keys for the metrics
from .keys import *
from .distances import _haversine, point_distance_metrics
## Point-Level Utility Metrics (Preservation)
from .distribution import JSD, sliced_wasserstein_distance, JS_from_points
## Point-Level Utility Metrics (Statistics)
from .point_statistics import hotspot_analysis_error, range_query_error, get_random_query_points
## Trajectory-Level Preservation Utility Metrics
from .trajectory_preservation import (hausdorff_distance, batched_trajectory_distance_minimized,
                                      batched_trajectory_distance, dtw_distance,
                                      haversine_distance, euclidean_distance, normalised_haversine_distance,
                                      normalised_euclidean_distance, reorder_dataset_to_matching)
## Trajectory-Level Statistics Utility Metrics
from .trajectory_statistics import trajectory_statistics_error, trip_error
## Integration of all metrics
from .integration import compute_all_metrics, save_metrics, load_metrics, load_case_metrics

