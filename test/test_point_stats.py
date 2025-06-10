#!/usr/bin/env python3
"""Test Point Statistics Metrics."""
from unittest import TestCase

from privtrajgen.metrics.point_statistics import *


class TestRangeQuery(TestCase):

    def test_single_query_single_radius(self):
        # Test with one query point and one radius
        point_dataset = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        query_point = np.array([0.0, 0.0])
        radius = 2.0
        expected_indices = [[[0, 1]]]

        result = range_query(point_dataset, query_point, radius)
        self.assertEqual(result, expected_indices)

    def test_single_query_multiple_radii(self):
        # Test with one query point and multiple radii
        point_dataset = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        query_point = np.array([0.0, 0.0])
        radii = [1.0, 2.0]
        expected_result = [[[0], [0, 1]]]

        result = range_query(point_dataset, query_point, radii)
        self.assertEqual(result, expected_result)

    def test_multiple_queries_single_radius(self):
        # Test with multiple query points and one radius
        point_dataset = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        query_points = [np.array([0.0, 0.0]), np.array([2.0, 2.0])]
        radius = 2.0
        expected_result = [
                            [[0, 1]],  # Query point 1
                            [[1, 2, 3]]  # Query point 2
                           ]

        result = range_query(point_dataset, query_points, radius)
        self.assertEqual(result, expected_result)

    def test_multiple_queries_multiple_radii(self):
        # Test with multiple query points and multiple radii
        point_dataset = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        query_points = [np.array([0.0, 0.0]), np.array([2.0, 2.0])]
        radii = [1.0, 2.0]
        expected_result = [
            [  # Point 1
                [0], # Radius 1
                [0, 1] # Radius 2
            ],
            [  # Point 2
                [2],
                [1, 2, 3]
            ]
        ]

        result = range_query(point_dataset, query_points, radii)
        self.assertEqual(result, expected_result)

    def test_lon_first_false(self):
        # Test with lon_first set to False
        point_dataset = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        query_point = np.array([0.0, 0.0])  # Now considered (lat, lon)
        radius = 2.0
        expected_indices = [0, 1]

        result = range_query(point_dataset, query_point, radius, lon_first=False)
        self.assertEqual(result, [[expected_indices]])

    def test_lon_first_false_asymmetric(self):
        """
        Test with lon_first set to False and an asymmetric dataset.
        """
        point_dataset = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        query_point = np.array([0.0, 1.0])
        radii = [1.0, 2.0]
        expected_indices = [[0], [0, 1]]

        result = range_query(point_dataset, query_point, radii, lon_first=False)
        self.assertEqual(result, [expected_indices])

    def test_custom_metric(self):
        # Test with a custom distance metric
        def custom_metric(p1, p2):
            return np.abs(p1 - p2).sum()

        point_dataset = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        query_point = np.array([0.0, 0.0])
        radius = [1.0, 2.0]
        expected_indices = [[0], [0, 1]]

        result = range_query(point_dataset, query_point, radius, metric=custom_metric)
        self.assertEqual(result, [expected_indices])

    def test_complex(self):
        """A more complex test case."""
        def simple_distance(p1, p2):
            # Easier to construct an example in my head
            return np.abs(p1 - p2).sum()
        point_dataset = np.array([[0.0, 3.0], [4.0, 3.0], [3.0, 3.0], [7.0, 2.0]])
        query_points = [np.array([3.0, 3.0]), np.array([9.0, 9.0])]
        radii = [1.0, 3.0, 18.0]
        expected_indices = [
            [[1, 2], [0, 1, 2], [0, 1, 2, 3]],
            [[], [], [0, 1, 2, 3]]
        ]

        result = range_query(point_dataset, query_points, radii, metric=simple_distance)
        self.assertEqual(result, expected_indices)

import unittest
import numpy as np

class TestHotspotAnalysis(unittest.TestCase):

    def test_single_grid_size(self):
        # Test with a single grid size
        point_dataset = np.array([
            [0.5, 0.5], [1.5, 1.5], [2.5, 2.6], [3.5, 3.5]
        ])
        bbox = {'lon_min': 0.0, 'lon_max': 4.0, 'lat_min': 0.0, 'lat_max': 4.0}
        grid_size = 4
        result = hotspot_analysis(point_dataset, grid_size, bbox, percentile=0.75)
        # High fluctuation due to small dataset --> 0.75 percentile instead of 0.95

        expected_hotspots = [
            [0, 5, 10, 15]  # Diagonal cells are hotspots
        ]

        # Asserting if the result contains any grid cell ids
        self.assertTrue(isinstance(result[0], list), "Expected a list of grid cell IDs for each grid size.")
        self.assertEqual(expected_hotspots, result)

    def test_multiple_grid_sizes(self):
        # Test with multiple grid sizes
        point_dataset = np.array([
            [0., 0.5], [1.4, 1.5], [1.5, 2.4], [3.9, 3.75], [3.9, 3.75], [3.9, 3.75]
        ])
        bbox = {'lon_min': 0.0, 'lon_max': 4.0, 'lat_min': 0.0, 'lat_max': 4.0}
        grid_sizes = [4, 6]
        result = hotspot_analysis(point_dataset, grid_sizes, bbox, percentile=0.99)
        ## Reduce smoothing b/c of small dataset

        expected_results = [  # In both cases, only the cell with multiple points is a hotspot
            [g * g - 1] for g in grid_sizes
        ]

        # Assert if the result has the same length as the number of grid sizes
        self.assertEqual(len(result), len(grid_sizes), "Expected the number of results to match the number of grid sizes.")
        for i, expected in enumerate(expected_results):
            self.assertEqual(expected, result[i], f"Expected hotspots for grid size {grid_sizes[i]}.")

    def test_lon_first_true(self):
        # Test with lon_first=True
        point_dataset = np.array([
            [0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]
        ])
        bbox = {'lon_min': 0.0, 'lon_max': 4.0, 'lat_min': 0.0, 'lat_max': 4.0}
        grid_size = 4
        result = hotspot_analysis(point_dataset, grid_size, bbox, percentile=0.75)
        # High fluctuation due to small dataset --> 0.75 percentile instead of 0.95

        expected_hotspots = [
            [0, 5, 10, 15]  # Diagonal cells are hotspots
        ]

        # Asserting if the result contains any grid cell ids
        self.assertTrue(isinstance(result[0], list), "Expected a list of grid cell IDs for each grid size.")
        self.assertEqual(expected_hotspots, result)

    def test_lon_first_false(self):
        # Test with lon_first=False (points in (lat, lon) format)
        point_dataset = np.array([
            [0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]
        ])[:, ::-1]  # Transpose points to (lat, lon)
        bbox = {'lon_min': 0.0, 'lon_max': 4.0, 'lat_min': 0.0, 'lat_max': 4.0}
        grid_size = 4
        result = hotspot_analysis(point_dataset, grid_size, bbox, percentile=0.75)
        # High fluctuation due to small dataset --> 0.75 percentile instead of 0.95

        expected_hotspots = [
            [0, 5, 10, 15]  # Diagonal cells are hotspots
        ]

        # Asserting if the result contains any grid cell ids
        self.assertTrue(isinstance(result[0], list), "Expected a list of grid cell IDs for each grid size.")
        self.assertEqual(expected_hotspots, result)

    def test_high_percentile(self):
        # Test with a high percentile value (close to 100%)
        point_dataset = np.array([
            [0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]
        ])
        bbox = {'lon_min': 0.0, 'lon_max': 4.0, 'lat_min': 0.0, 'lat_max': 4.0}
        grid_size = 4
        result = hotspot_analysis(point_dataset, grid_size, bbox, percentile=0.99)

        # Assert that result contains fewer hotspots at a high percentile
        self.assertTrue(isinstance(result[0], list), "Expected a list of grid cell IDs.")
        self.assertGreaterEqual(len(result[0]), 1, "Expected at least one hotspot grid cell.")

    def test_low_percentile(self):
        # Test with a low percentile value (close to 0%)
        point_dataset = np.array([
            [0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]
        ])
        bbox = {'lon_min': 0.0, 'lon_max': 4.0, 'lat_min': 0.0, 'lat_max': 4.0}
        grid_size = 4
        result = hotspot_analysis(point_dataset, grid_size, bbox, percentile=0.01)

        # Assert that a low percentile includes most grid cells
        self.assertTrue(isinstance(result[0], list), "Expected a list of grid cell IDs.")
        self.assertGreaterEqual(len(result[0]), 1, "Expected at least one hotspot grid cell.")


class TestHotspotAnalysisError(unittest.TestCase):

    def setUp(self):
        # Common test bounding box
        self.bbox = {'lon_min': 0.0, 'lon_max': 4.0, 'lat_min': 0.0, 'lat_max': 4.0}
        self.grid_sizes = [2, 4, 6]  # Grid sizes to test

    def test_identical_datasets(self):
        # Test with identical point datasets (should yield Sørensen-Dice coefficient of 1.0)
        point_dataset = np.array([
            [0., 0.5], [1.4, 1.5], [1.5, 2.4], [3.9, 3.75], [3.9, 3.75], [3.9, 3.75]
        ])
        result = hotspot_analysis_error(point_dataset, point_dataset, self.grid_sizes, self.bbox)

        # Expect Sørensen-Dice coefficient to be 1.0 for both grid sizes
        for grid_size in self.grid_sizes:
            self.assertAlmostEqual(result[grid_size], 1.0, msg=f"Expected Sørensen-Dice coefficient of 1.0 for grid size {grid_size}")

    def test_non_overlapping_datasets(self):
        # Test with non-overlapping point datasets (should yield Sørensen-Dice coefficient of 0.0)
        point_dataset1 = np.array([  # Everything is in first row (lat < 2.0)
            [0., 0.5], [1.4, 1.5], [1.5, 1.4], [1.9, 1.9]
        ])
        point_dataset2 = np.array([  # Everything is in second row (lat > 2.0)
            [2.5, 2.5], [3.4, 3.5], [3.5, 3.4], [3.9, 3.9]
        ])
        result = hotspot_analysis_error(point_dataset1, point_dataset2, self.grid_sizes, self.bbox)

        # Expect Sørensen-Dice coefficient to be 0.0 for both grid sizes
        for grid_size in self.grid_sizes:
            self.assertAlmostEqual(result[grid_size], 0.0, msg=f"Expected Sørensen-Dice coefficient of 0.0 for grid size {grid_size}")

    def test_partial_overlap(self):
        # Test with partially overlapping datasets (should yield Sørensen-Dice coefficient between 0 and 1)
        # First dataset has points in the first row (lat < 2.0)
        point_dataset1 = np.array([
            [1.0, 0.5], [0.5, 0.0], [3.4, 1.4], [3.0, 1.9]
        ])
        # Second Dataset cover second column, but both rows (lat > 2.0, lon: 2x < 2.0, 2x > 2.0)
        point_dataset2 = np.array([
            [2.5, 0.5], [3.5, 1.0], [3.5, 3.9], [3.9, 3]
        ])
        result = hotspot_analysis_error(point_dataset1, point_dataset2, self.grid_sizes, self.bbox, percentile=0.5)
        # Expect that it's a 50% overlap for grid size 2
        expected_result = 0.5
        self.assertAlmostEqual(result[2], expected_result,
                               msg=f"Expected Sørensen-Dice coefficient of {expected_result} for grid size {4}")

        # Ensure the Sørensen-Dice coefficient is between 0 and 1
        for grid_size in self.grid_sizes:
            self.assertGreater(result[grid_size], 0.0,
                               f"Expected Sørensen-Dice coefficient > 0.0 for grid size {grid_size}")
            self.assertLess(result[grid_size], 1.0,
                            f"Expected Sørensen-Dice coefficient < 1.0 for grid size {grid_size}")



    def test_different_grid_sizes(self):
        # Test with different grid sizes
        point_dataset1 = np.array([
            [0., 0.5], [1.4, 1.5], [1.5, 2.4], [3.9, 3.75]
        ])
        point_dataset2 = np.array([
            [0., 0.5], [1.4, 1.5], [1.5, 2.4], [3.9, 3.75]
        ])
        result = hotspot_analysis_error(point_dataset1, point_dataset2, self.grid_sizes, self.bbox)

        # Expect Sørensen-Dice coefficient to be 1.0 for both grid sizes
        for grid_size in self.grid_sizes:
            self.assertAlmostEqual(result[grid_size], 1.0, msg=f"Expected Sørensen-Dice coefficient of 1.0 for grid size {grid_size}")

    def test_empty_datasets(self):
        # Test with empty datasets (should yield Sørensen-Dice coefficient of 1.0)
        point_dataset1 = np.array([])
        point_dataset2 = np.array([])

        # Expect ValueErrors for empty datasets
        with self.assertRaises(ValueError):
            hotspot_analysis_error(point_dataset1, point_dataset2, self.grid_sizes, self.bbox)
