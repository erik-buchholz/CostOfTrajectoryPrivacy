#!/usr/bin/env python3
"""Keys for the Utility Metrics."""
from enum import Enum

METADATA = "metadata"

JSD_DISTRIBUTION = "JSD_Distribution"
SWD_DISTRIBUTION = "SWD_Distribution"
HD_POINTS = "HD_Points"
POINT_PRESERVATION_METRICS = [JSD_DISTRIBUTION, SWD_DISTRIBUTION, HD_POINTS]

HOTSPOT_ANALYSIS = "Hotspot_Analysis"
RANGE_QUERY = "Range_Query"
MAE = 'MAE'
MRE = 'MRE'
DATASET_1 = 'Dataset_1'  # The dataset that is being compared to
DATASET_2 = 'Dataset_2'  # The dataset that is being compared
POINT_STATISTICS_METRICS = [HOTSPOT_ANALYSIS, RANGE_QUERY]


# Trajectory Distance Metrics - per trajectory
HD_TRAJECTORIES = "HD_Trajectories"
ED_TRAJECTORIES = "ED_Trajectories"
DTW_TRAJECTORIES = "DTW_Trajectories"
TRAJ_PRESERVATION_METRICS_EXACT= [HD_TRAJECTORIES, ED_TRAJECTORIES, DTW_TRAJECTORIES]

# Trajectory Distance Metrics - aggregated
HD_AGGREGATED = "HD_Aggregated"
ED_AGGREGATED = "ED_Aggregated"
DTW_AGGREGATED = "DTW_Aggregated"
TRAJ_PRESERVATION_METRICS_AGGREGATED = [HD_AGGREGATED, ED_AGGREGATED, DTW_AGGREGATED]
# These will be dicts that contain:
MEAN = "Mean"
STD = "Std"
NUM = "Num"
SUM = "Sum"

# Matching dict assigning each metric key to the aggregation key
TRAJ_METRIC_AGGREGATION_MAPPING = {
    HD_TRAJECTORIES: HD_AGGREGATED,
    ED_TRAJECTORIES: ED_AGGREGATED,
    DTW_TRAJECTORIES: DTW_AGGREGATED
}
TRAJ_PRES_EXTENSION = "_traj_preservation.npz"

TTD = "TTD"
DIAMETER = "Diameter"
TRIP_ERROR = "Trip_Error"
TRAJ_STATISTICS_METRICS = [TTD, DIAMETER, TRIP_ERROR]

ALL_METRICS = [
    JSD_DISTRIBUTION,
    SWD_DISTRIBUTION,
    HD_POINTS,
    HOTSPOT_ANALYSIS,
    RANGE_QUERY,
    HD_TRAJECTORIES,
    ED_TRAJECTORIES,
    DTW_TRAJECTORIES,
    HD_AGGREGATED,
    ED_AGGREGATED,
    DTW_AGGREGATED,
    TTD,
    DIAMETER,
    TRIP_ERROR
]


class PointDistanceMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    HAVERSINE = "haversine"

    def __str__(self) -> str:
        return self.value

# Create Enum of Trajectory Distance Metrics
# All distance metrics have to accept the lon_first flat to ensure compatibility with the other metrics.
# if a metric does not, it has to be wrapped in a function that accepts lon_first as an argument.
class TrajectoryDistanceMetric(str, Enum):
    HAUSDORFF = "hausdorff"
    DTW = "dtw"
    EUCLIDEAN = "euclidean"
    HAVERSINE = "haversine"

    def __str__(self) -> str:
        return self.value
