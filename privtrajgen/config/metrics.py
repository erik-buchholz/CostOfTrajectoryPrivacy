#!/usr/bin/env python3
"""Evaluation Configuration.

For the below runtime notes, we used the following evaluation hardware:
OS: Ubuntu 24.04.1 LTS x86_64
CPU: AMD EPYC 7763 (64) @ 2.450GHz
Memory: 540G GB
GPU: NVIDIA RTX A6000 (48GB) x 2

The following references are used for the configuration:
[1] T. Cunningham, K. Klemmer, H. Wen, and H. Ferhatosmanoglu,
“GeoPointGAN: Synthetic Spatial Data with Local Label Differential Privacy,” May 18, 2022.
doi: 10.48550/arXiv.2205.08886.
[2] R. Chen, B. C. M. Fung, and B. C. Desai,
“Differentially Private Trajectory Data Publication,” Dec. 09, 2011. doi: 10.48550/arXiv.1112.2020.
[3] Y. Zhu, Y. Ye, S. Zhang, X. Zhao, and J. J. Q. Yu,
“DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model,”
in Advances in Neural Information Processing Systems, in 1, vol. 23.
New Orleans, USA: Curran Associates, Inc., 2023, p. 21. doi: 10.5555/3666122.3668965.
[4] X. Sun et al., “Synthesizing realistic trajectory data with differential privacy,”
IEEE Trans. Intell. Transp. Syst., vol. 24, no. 5, pp. 5502–5515, May 2023, doi: 10.1109/TITS.2023.3241290.
"""
from .enums import JSDModes

########################## General Configuration #####################################
EVALUATION_TRAJECTORY_NUM = 3000  # Number of trajectories to generate. Same as [3].
NORMALISE_TRAJECTORY_DISTANCES = True  # Normalise each per-trajectory distance by the trajectory length
JSD_MODE_STATISTICS = JSDModes.DIVERGENCE  # Use the JS distance for the statistics-based metrics
POINT_LIMIT = None  # Limit the number of points in the dataset. None for no limit. Due to overhead.
TRAJECTORY_LIMIT = None  # Limit the number of trajectories in the dataset. None for no limit. Due to overhead.

########################## Point Distribution Configuration ##########################
JSD_GRID_SIZES = [16, 32, 64]  # Based on [3], added 64 for more granularity.
JSD_MODE_DISTRIBUTION = JSDModes.DIVERGENCE  # Use the JS divergence for the distribution-based metrics
# for consistency with [3].
HD_POINT_LIMIT = 50000  # The HD gets computationally expensive with large datasets & allocated a lot of memory.
# Note that we *randomly* sample from all points in the dataset.
# 100k points takes ~5min and >400GB of memory on our machine.

########################## Range Query Error Configuration ###########################
# Sanity Bound Factor is used for the relative error calculation in the range query error.
SANITY_BOUND_FACTOR = 0.001  # Use 0.1% of the dataset size. Based on [2]
NUM_RANGE_QUERIES = 200 # Number of query points for the range query error calculation. Based on [1]
RANGE_QUERY_RADII = [50, 100, 200, 500, 1000]  # in meter. Based on [1].

########################## Hotspot Analysis Configuration ############################
HOTSPOT_PERCENTILE = 0.95  # Based on [1]
HOTSPOT_GRID_SIZES = [16, 32, 64, 128]  # [64, 128, 256, 512, 1024]
# [1] uses (2^6, 2^7, 2^8, 2^9, 2^10) = (64, 128, 256, 512, 1024) grid sizes for hotspot analysis.
# However, this metric is the bottleneck in the evaluation, so we use smaller grid sizes.
# The configuration up to 256 takes ~13min on our machine.

########################## Trip Error Configuration ##################################
TRIP_ERROR_GRID_SIZE = 16  # Based on [3], could not find any concrete values used in literature.
# However, the transition matrix has size grid_size^2^2 = grid_size^4 = 16^4 = 65,536, which is reasonably large.

########################## Trajectory Statistics Configuration ########################
N_BINS_TRAJ_STATS = 55  # [4] used 20 bins. However, use the square root of the number of trajectories as a
# rule of thumb for the number of bins. We have 3000 trajectories, so use sqrt(3000) = 54.77 ~ 55 bins.
