#!/usr/bin/env python3
"""Test Trajectory Statistics Metrics."""
from unittest import TestCase

from privtrajgen.metrics.trajectory_statistics import *


class TestTrajectoryStatistics(TestCase):

    def test_total_travelled_distance_euclidean(self):
        # Test total travelled distance with Euclidean metric
        traj = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        total_dist = total_travelled_distance(traj, metric=PointDistanceMetric.EUCLIDEAN)
        expected_dist = np.linalg.norm([3.0, 4.0]) * 2
        self.assertAlmostEqual(total_dist, expected_dist)

    def test_total_travelled_distance_custom_metric(self):
        # Test total travelled distance with a custom metric
        def custom_metric(p1, p2):
            return np.abs(p1 - p2).sum()

        traj = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        total_dist = total_travelled_distance(traj, metric=custom_metric)
        expected_dist = 7 * 2  # |3-0| + |4-0| = 7, happens twice
        self.assertAlmostEqual(total_dist, expected_dist)

    def test_trajectory_diameter_euclidean(self):
        # Test trajectory diameter with Euclidean metric
        traj = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        diameter = trajectory_diameter(traj, metric=PointDistanceMetric.EUCLIDEAN)
        expected_diameter = np.linalg.norm([6.0, 8.0])  # distance between first and last point
        self.assertAlmostEqual(diameter, expected_diameter)

    def test_trajectory_diameter_custom_metric(self):
        # Test trajectory diameter with a custom metric
        def custom_metric(p1, p2):
            return np.abs(p1 - p2).sum()

        traj = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        diameter = trajectory_diameter(traj, metric=custom_metric)
        expected_diameter = 14  # |6-0| + |8-0| = 14
        self.assertAlmostEqual(diameter, expected_diameter)

    def test_trajectory_statistics_euclidean(self):
        # Test trajectory statistics with Euclidean metric
        traj = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        stats = trajectory_statistics(traj, metric=PointDistanceMetric.EUCLIDEAN)
        expected_total_distance = np.linalg.norm([3.0, 4.0]) + np.linalg.norm([3.0, 4.0])
        expected_diameter = np.linalg.norm([6.0, 8.0])
        self.assertAlmostEqual(stats[TTD], expected_total_distance)
        self.assertAlmostEqual(stats[DIAMETER], expected_diameter)

    def test_trajectory_statistics_custom_metric(self):
        # Test trajectory statistics with a custom metric
        def custom_metric(p1, p2):
            return np.abs(p1 - p2).sum()

        traj = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        stats = trajectory_statistics(traj, metric=custom_metric)
        expected_total_distance = 7 * 2  # |3-0| + |4-0| = 7, happens twice
        expected_diameter = 14  # |6-0| + |8-0| = 14
        self.assertAlmostEqual(stats[TTD], expected_total_distance)
        self.assertAlmostEqual(stats[DIAMETER], expected_diameter)


class TestTripError(TestCase):

    def test_transition_matrices(self):
        # Test transition matrices with a simple trajectory dataset
        real_traj_dataset = np.array([
            [[0.0, 0.0], [3.0, 4.0]],
            [[6.0, 8.0], [0.0, 0.0]]
        ])
        grid_size = 2
        bbox = {'lat_min': 0.0, 'lat_max': 10.0, 'lon_min': 0.0, 'lon_max': 10.0}

        transition_matrix = transition_matrices(real_traj_dataset, grid_size, bbox)
        expected_shape = (grid_size * grid_size, grid_size * grid_size)
        expected_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0]
        ]) / len(real_traj_dataset)

        self.assertEqual(transition_matrix.shape, expected_shape)
        np.testing.assert_array_equal(transition_matrix, expected_matrix)

    def test_transition_matrices2(self):
        # Test transition matrices with a simple trajectory dataset
        # 0-5 is first cell and 6-10 is second cell
        real_traj_dataset = np.array([
            [[0.0, 0.0], [10.0, 10.0]],  # 0 -> 3
            [[1.0, 1.0], [9.0, 9.0]],  # 0 -> 3
            [[4.0, 3.0], [4.0, 8.0]],  # 0 -> 2
            [[6.0, 0.0], [0.0, 6.0]], # 1 -> 2
            [[7.0, 9.0], [1.0, 7.0]], # 3 -> 2
            [[0.0, 7.0], [10.0, 3.0]], # 2 -> 1
        ])
        grid_size = 2
        bbox = {'lat_min': 0.0, 'lat_max': 10.0, 'lon_min': 0.0, 'lon_max': 10.0}

        transition_matrix = transition_matrices(real_traj_dataset, grid_size, bbox)
        expected_shape = (grid_size * grid_size, grid_size * grid_size)
        expected_matrix = np.array([
            [0, 0, 1, 2],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]) / len(real_traj_dataset)

        self.assertEqual(transition_matrix.shape, expected_shape)
        np.testing.assert_array_equal(transition_matrix, expected_matrix)

    def test_trip_error_identical_datasets(self):
        # Test trip error with identical datasets
        real_traj_dataset = np.array([
            [[0.0, 0.0], [3.0, 4.0]],
            [[6.0, 8.0], [0.0, 0.0]]
        ])
        gen_traj_dataset = real_traj_dataset.copy()
        grid_size = 2
        bbox = {'lat_min': 0.0, 'lat_max': 10.0, 'lon_min': 0.0, 'lon_max': 10.0}

        error = trip_error(real_traj_dataset, gen_traj_dataset, grid_size, bbox)
        self.assertAlmostEqual(error, 0.0)

    def test_trip_error_different_datasets(self):
        # Test trip error with different datasets
        real_traj_dataset = np.array([
            [[0.0, 0.0], [3.0, 4.0]],
            [[6.0, 8.0], [0.0, 0.0]]
        ])
        gen_traj_dataset = np.array([
            [[1.0, 1.0], [4.0, 5.0]],
            [[7.0, 9.0], [1.0, 1.0]]
        ])
        grid_size = 2
        bbox = {'lat_min': 0.0, 'lat_max': 10.0, 'lon_min': 0.0, 'lon_max': 10.0}

        error = trip_error(real_traj_dataset, gen_traj_dataset, grid_size, bbox)
        self.assertGreater(error, 0.0)