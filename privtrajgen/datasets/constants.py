#!/usr/bin/env python3
"""Constants used for the datasets."""
from privtrajgen.config import FeatureType, LAT, LON, FSNYC, GEOLIFE, PORTO

VOCAB_SIZE = {
    'latlon': 2,
    'hour': 24,
    'day': 7,
    'category': 10,
}
EMBEDDING_SIZE = {
    'latlon': 64,
    'hour': 24,
    'day': 7,
    'category': 10
}

FEATURE_TYPES = {
    'latlon': FeatureType.CONTINUOUS,
    'hour': FeatureType.CATEGORICAL,
    'day': FeatureType.CATEGORICAL,
    'category': FeatureType.CATEGORICAL,
    'mnist': FeatureType.CONTINUOUS,
}

# DiffTraj Conditional Information
COND_ATTRIBUTES_DiffTraj = ["dep_time", "total_dist", "total_dur", "len", "avg_dist", "avg_speed", "sid", "eid"]
COND_ATTRIBUTE_NAMES_DiffTraj = ["Departure Time", "Total Distance", "Total Duration", "Total Length", "Avg Distance",
                                 "Average Speed", "Start Grid ID", "End Grid ID"]
NORMALISED_ATTRIBUTES_DiffTraj = ["total_dist", "total_dur", "len", "avg_dist", "avg_speed"]
CATEGORICAL_ATTRIBUTES_DiffTraj = ["dep_time", "sid", "eid"]

##############################################################################
# Foursquare NYC
##############################################################################
NUM_PROCESSED_TRAJS_FS = 3079
MAX_LEN_FS = 144
REF_FS = {LAT: 40.760000000000005, LON: -73.975}
SCALE_FS = {LAT: 0.240000000000002, LON: 0.32500000000000284}

##############################################################################
# Porto
##############################################################################
NUM_PROCESSED_TRAJS_PORTO = 1559209
MAX_LEN_PORTO = 100
REF_POINT_PORTO = {LAT: 41.1700005, LON: -8.6099985}
SCALE_FACTOR_PORTO = {LAT: 0.0699975, LON: 0.1099935}

##############################################################################
# GeoLife
##############################################################################
NUM_PROCESSED_TRAJS_GEOLIFE = 69504  # See geoLife.ipynb
MAX_LEN_GEOLIFE = 200
GL_REF_POINT_95 = {LAT: 35.60931733333335,
                   LON: 113.2185733333335}
GL_SCALE_FACTOR_95 = {LAT: 4.76171566666665,
                      LON: 4.748456666666499}
GL_REF_POINT_FIFTH = {LAT: 39.89000165,
                      LON: 116.37499995}
GL_SCALE_FACTOR_FIFTH = {LAT: 0.13999835,
                         LON: 0.18499995}
DEFAULT_REF_POINT_GEOLIFE = GL_REF_POINT_FIFTH
DEFAULT_SCALE_FACTOR_GEOLIFE = GL_SCALE_FACTOR_FIFTH

##############################################################################
# Structs
##############################################################################
ref_points = {
    FSNYC: REF_FS,
    GEOLIFE: DEFAULT_REF_POINT_GEOLIFE,
    PORTO: REF_POINT_PORTO,
}
scale_factors = {
    FSNYC: SCALE_FS,
    GEOLIFE: DEFAULT_SCALE_FACTOR_GEOLIFE,
    PORTO: SCALE_FACTOR_PORTO,
}
trajectory_lengths = {
    FSNYC: MAX_LEN_FS,
    GEOLIFE: MAX_LEN_GEOLIFE,
    PORTO: MAX_LEN_PORTO,
}
number_of_trajectories = {
    FSNYC: NUM_PROCESSED_TRAJS_FS,
    GEOLIFE: NUM_PROCESSED_TRAJS_GEOLIFE,
    PORTO: NUM_PROCESSED_TRAJS_PORTO,
}
