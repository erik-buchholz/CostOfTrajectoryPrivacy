#!/usr/bin/env python3
import json
import logging
import unittest

import numpy as np

from privtrajgen.config import *
from privtrajgen.metrics import compute_all_metrics, save_metrics, load_metrics
from privtrajgen.metrics.keys import *
from privtrajgen.utils.logger import configure_root_loger

# Activate Debug Logging
configure_root_loger(logging.INFO,error_file="error_unittest.log")

def scale_to_bbox(trajectory_dataset, bbox):
    """
    Scale the trajectory dataset into the bounding box.
    """
    lon_min = bbox['lon_min']
    lon_max = bbox['lon_max']
    lat_min = bbox['lat_min']
    lat_max = bbox['lat_max']

    lon = trajectory_dataset[:, :, 0]
    lat = trajectory_dataset[:, :, 1]

    lon_scaled = lon - np.min(lon)
    lon_scaled /= np.max(lon_scaled)
    lon_scaled *= (lon_max - lon_min)
    lon_scaled += lon_min

    lat_scaled = lat - np.min(lat)
    lat_scaled /= np.max(lat_scaled)
    lat_scaled *= (lat_max - lat_min)
    lat_scaled += lat_min

    return np.stack([lon_scaled, lat_scaled], axis=2)


class TestMetricIntegration(unittest.TestCase):

    def setUp(self):
        # Initialize mock trajectory datasets and parameters
        self.lon_first = True
        self.geographical_coordinates = False
        self.are_matched = False
        self.bbox = {'lon_min': 0.0, 'lon_max': 4.0, 'lat_min': 0.0, 'lat_max': 4.0}
        self.trajectory_dataset1 = scale_to_bbox(np.random.randn(100, 10, 2), self.bbox)
        self.trajectory_dataset2 = scale_to_bbox(np.random.randn(100, 10, 2), self.bbox)
        # hotspot analysis is bottleneck --> Use smaller grid sizes for unittests
        self.hotspot_grid_sizes = [16, 32]
        self.radii = [0.2, 0.4, 0.6, 0.8, 1.0]


    def test_compute_all_metrics(self):
        results = compute_all_metrics(
            trajectory_dataset1=self.trajectory_dataset1,
            trajectory_dataset2=self.trajectory_dataset2,
            lon_first=self.lon_first,
            geographical_coordinates=self.geographical_coordinates,
            are_matched=self.are_matched,
            bbox=self.bbox,
            hotspot_grid_sizes=self.hotspot_grid_sizes,
            range_query_radii=self.radii
        )

        # Assert that each metric is computed and has the correct type
        self.assertIsInstance(results[JSD_DISTRIBUTION], dict)
        for grid_size in JSD_GRID_SIZES:
            self.assertIsInstance(results[JSD_DISTRIBUTION][grid_size], float)
        self.assertIsInstance(results[SWD_DISTRIBUTION], float)
        self.assertIsInstance(results[HD_POINTS], float)
        self.assertIsInstance(results[RANGE_QUERY], dict)
        self.assertIsInstance(results[RANGE_QUERY][MAE], float)
        self.assertIsInstance(results[RANGE_QUERY][MRE], float)
        self.assertIsInstance(results[HOTSPOT_ANALYSIS], dict)
        for grid_size in self.hotspot_grid_sizes:
            self.assertIsInstance(results[HOTSPOT_ANALYSIS][grid_size], float)
        # Trajectory Preservation Metrics
        for metric in TRAJ_METRIC_AGGREGATION_MAPPING.keys():
            self.assertIsInstance(results[metric], np.ndarray)
            self.assertEqual(results[metric].shape, (len(self.trajectory_dataset1),))
        # Aggregated Trajectory Preservation Metrics
        for metric in TRAJ_METRIC_AGGREGATION_MAPPING.values():
            self.assertIsInstance(results[metric], dict)
            self.assertIsInstance(results[metric][MEAN], float)
            self.assertIsInstance(results[metric][STD], float)
            self.assertIsInstance(results[metric][NUM], int)
            self.assertIsInstance(results[metric][SUM], float)
        self.assertIsInstance(results[TRIP_ERROR], float)
        self.assertIsInstance(results[TTD], float)
        self.assertIsInstance(results[DIAMETER], float)

        # Test saving to file
        tmp_dir = config.RESULT_DIR
        save_metrics(results, 'unittest.json',
                     metadata={'Unittest': 'Metric Integration Test'},
                     output_dir=tmp_dir, per_traj_metrics=True)

        expected_filename = Path(tmp_dir).joinpath('unittest.json')
        expected_exact_filename = Path(tmp_dir).joinpath('unittest' + TRAJ_PRES_EXTENSION)

        # Check that the files was saved
        self.assertTrue(expected_filename.exists())
        self.assertTrue(expected_exact_filename.exists())


        # Load the saved JSON file
        with open(expected_filename, 'r') as f:
            loaded_metrics = json.load(f)

        # Assert that the loaded metrics are the same as the computed metrics (except the exact preservation metrics)
        for key in results:
            if key not in TRAJ_PRESERVATION_METRICS_EXACT:
                if isinstance(results[key], dict):
                    # Ignore that the keys in the loaded dict are strings
                    new_dict = {str(k): v for k, v in results[key].items()}
                    self.assertEqual(new_dict, loaded_metrics[key], f"Key: {key}")
                else:
                    self.assertEqual(results[key], loaded_metrics[key], f"Key: {key}")

        # Load the exact preservation metrics (from npz file)
        loaded_exact_metrics = np.load(expected_exact_filename)

        # Assert that the loaded exact preservation metrics are the same as the computed metrics
        for key in TRAJ_PRESERVATION_METRICS_EXACT:
            self.assertTrue(np.allclose(results[key], loaded_exact_metrics[key]), f"Key: {key}")

        # Test the load function
        loaded_results = load_metrics('unittest.json', per_traj_metrics=True)

        # Assert that the loaded metrics are the same as the computed metrics
        for key in results:
            if isinstance(results[key], dict):
                # Ignore that the keys in the loaded dict are strings
                new_dict = {str(k): v for k, v in results[key].items()}
                if isinstance(loaded_results[key], np.ndarray):
                    self.assertTrue(np.allclose(new_dict, loaded_results[key]), f"Key: {key}")
                else:
                    self.assertEqual(new_dict, loaded_results[key], f"Key: {key}")
            else:
                if isinstance(loaded_results[key], np.ndarray):
                    self.assertTrue(np.allclose(results[key], loaded_results[key]), f"Key: {key}")
                else:
                    self.assertEqual(results[key], loaded_results[key], f"Key: {key}")





    def test_identical_sets(self):
        results = compute_all_metrics(
            trajectory_dataset1=self.trajectory_dataset1,
            trajectory_dataset2=self.trajectory_dataset1,  # identical datasets
            lon_first=self.lon_first,
            geographical_coordinates=self.geographical_coordinates,
            are_matched=self.are_matched,
            bbox=self.bbox,
            hotspot_grid_sizes=self.hotspot_grid_sizes,
            range_query_radii=self.radii
        )

        # Assert that each metric is computed and has the expected value for identical datasets

        # Jensen-Shannon Divergence should be 0 for identical datasets
        for grid_size in JSD_GRID_SIZES:
            self.assertAlmostEqual(results[JSD_DISTRIBUTION][grid_size], 0.0, places=6)

        # Sliced Wasserstein Distance should be 0 for identical datasets
        self.assertAlmostEqual(results[SWD_DISTRIBUTION], 0.0, places=6)

        # Hausdorff Distance should be 0 for identical datasets
        self.assertAlmostEqual(results[HD_POINTS], 0.0, places=6)

        # Range Query Errors (MAE and MRE) should be 0 for identical datasets
        self.assertAlmostEqual(results[RANGE_QUERY][MAE], 0.0, places=6)
        self.assertAlmostEqual(results[RANGE_QUERY][MRE], 0.0, places=6)

        # SDC(Hotspot Analysis) for identical datasets should be 1
        for grid_size in self.hotspot_grid_sizes:
            self.assertAlmostEqual(results[HOTSPOT_ANALYSIS][grid_size], 1.0, places=6)

        # Trajectory Preservation Metrics should be 0 for identical datasets
        for metric in TRAJ_METRIC_AGGREGATION_MAPPING:
            # The exact values per trajectory should be 0
            self.assertTrue(np.allclose(results[metric], np.zeros(len(self.trajectory_dataset1))))
            # Aggregated Trajectory Preservation Metrics
            agg_key = TRAJ_METRIC_AGGREGATION_MAPPING[metric]
            self.assertAlmostEqual(results[agg_key][MEAN], 0.0, places=6)
            self.assertAlmostEqual(results[agg_key][STD], 0.0, places=6)
            self.assertEqual(results[agg_key][NUM], len(self.trajectory_dataset1))
            self.assertAlmostEqual(results[agg_key][SUM], 0.0, places=6)

        # JSD(Trip Error) should be 0 for identical datasets
        self.assertAlmostEqual(results[TRIP_ERROR], 0.0, places=6)

        # JSD(TTD) (Total Trajectory Distance) should be 0 for identical datasets
        self.assertAlmostEqual(results[TTD], 0.0, places=6)

        # JSD(Diameter) should be 0 for identical datasets
        self.assertAlmostEqual(results[DIAMETER], 0.0, places=6)

    def test_differing_sets(self):
        results = compute_all_metrics(
            trajectory_dataset1=self.trajectory_dataset1,
            trajectory_dataset2=self.trajectory_dataset2,  # differing datasets
            lon_first=self.lon_first,
            geographical_coordinates=self.geographical_coordinates,
            are_matched=self.are_matched,
            bbox=self.bbox,
            hotspot_grid_sizes=self.hotspot_grid_sizes,
            range_query_radii=self.radii
        )

        # Assert that each metric is computed and has non-zero values for differing datasets

        # Jensen-Shannon Divergence should be greater than 0 for differing datasets
        for grid_size in JSD_GRID_SIZES:
            self.assertGreater(results[JSD_DISTRIBUTION][grid_size], 0.0)

        # Sliced Wasserstein Distance should be greater than 0 for differing datasets
        self.assertGreater(results[SWD_DISTRIBUTION], 0.0)

        # Hausdorff Distance should be greater than 0 for differing datasets
        self.assertGreater(results[HD_POINTS], 0.0)

        # Range Query Errors (MAE and MRE) should be greater than 0 for differing datasets
        self.assertGreater(results[RANGE_QUERY][MAE], 0.0)
        self.assertGreater(results[RANGE_QUERY][MRE], 0.0)

        # Hotspot Analysis should be less than 1 for differing datasets (SDC is expected to be lower for dissimilar sets)
        for grid_size in self.hotspot_grid_sizes:
            self.assertLess(results[HOTSPOT_ANALYSIS][grid_size], 1.0)

        # Trajectory Preservation Metrics should be greater than 0 for differing datasets
        for metric in TRAJ_METRIC_AGGREGATION_MAPPING:
            # The exact values per trajectory should be greater than 0
            self.assertTrue(np.any(results[metric] > 0.0))
            # Aggregated Trajectory Preservation Metrics
            agg_key = TRAJ_METRIC_AGGREGATION_MAPPING[metric]
            self.assertGreater(results[agg_key][MEAN], 0.0)
            self.assertGreater(results[agg_key][STD], 0.0)
            self.assertEqual(results[agg_key][NUM], len(self.trajectory_dataset1))
            self.assertGreater(results[agg_key][SUM], 0.0)

        # Trip Error should be greater than 0 for differing datasets
        self.assertGreater(results[TRIP_ERROR], 0.0)

        # Total Trajectory Distance (TTD) should be greater than 0 for differing datasets
        self.assertGreater(results[TTD], 0.0)

        # Diameter should be greater than 0 for differing datasets
        self.assertGreater(results[DIAMETER], 0.0)

    def tearDown(self):
        pass



if __name__ == '__main__':
    unittest.main()
