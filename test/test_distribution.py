#!/usr/bin/env python3
"""Test Distribution Metrics."""
from unittest import TestCase

import numpy as np
from scipy.spatial.distance import euclidean

from privtrajgen.metrics import *


# noinspection PyUnusedLocal
def euclidean_dist(traj1, traj2, **kwargs):
    return np.sum([euclidean(traj1[i], traj2[i]) for i, _ in enumerate(traj1)])


class TestMatching(TestCase):

    def setUp(self):
        # Example test data: simple trajectories
        self.batch1 = [
            np.array([[0, 0], [1, 1], [2, 2]]),  # Diagonal line
            np.array([[1, 1], [2, 2], [3, 3]]),  # Shifted diagonal line
        ]
        self.batch2 = [
            np.array([[0, 0], [1, 1], [2, 2]]),  # Same as first trajectory in batch1
            np.array([[1, 1], [2, 2], [3, 3]]),  # Same as second trajectory in batch1
        ]
        self.batch_different_size = [
            np.array([[0, 0], [1, 1], [2, 2]])  # Only one trajectory
        ]

    def test_matching_with_identical_batches(self):
        # Test with identical batches, should match each trajectory with itself
        matching, minimized_distances, total_distance = batched_trajectory_distance_minimized(
            self.batch1, self.batch2, euclidean_dist
        )
        self.assertEqual(matching, [(0, 0), (1, 1)])  # Expecting perfect match
        self.assertTrue(np.allclose(minimized_distances, [0.0, 0.0]))  # All distances should be zero
        self.assertAlmostEqual(total_distance, 0.0)

    def test_distance_computation(self):
        batch1 = [
            np.array([[0, 0], [1, 1], [2, 2]]),  # Diagonal line
            np.array([[1, 1], [2, 2], [3, 3]]),  # Shifted diagonal line
            np.array([[5, 5], [6, 6], [7, 7]]),  # Another diagonal line
        ]
        batch2 = [
            np.array([[1, 2], [2, 2], [3, 2]]),
            np.array([[0, 1], [1, 1], [2, 1]]),
            np.array([[5, 5], [7, 7], [7, 7]]),  # Another diagonal line
        ]
        expected_distances = [2, 2, np.sqrt(2)]
        expected_total_distance = np.sum(expected_distances)
        expected_matching = [(0, 1), (1, 0), (2, 2)]
        matching, minimized_distances, total_distance = batched_trajectory_distance_minimized(
            batch1, batch2, euclidean_dist)
        self.assertEqual(matching, expected_matching)
        self.assertTrue(np.allclose(minimized_distances, expected_distances))
        self.assertAlmostEqual(total_distance, expected_total_distance)

    def test_matching_with_shifted_batches(self):
        # Test with slightly shifted batches, should still match each trajectory to the closest
        batch_shifted = [
            np.array([[1, 0], [2, 1], [3, 2]]),  # Shifted version of the first trajectory
            np.array([[0, 0], [1, 1], [2, 2]]),  # Same as first trajectory in batch1
        ]
        matching, minimized_distances, total_distance = batched_trajectory_distance_minimized(
            self.batch1, batch_shifted, euclidean_dist
        )
        self.assertEqual(matching, [(0, 1), (1, 0)])  # Should match trajectories in the optimal way
        self.assertTrue(np.all(minimized_distances >= 0.0))  # Distances should be non-negative
        self.assertGreater(total_distance, 0.0)  # Total distance should be greater than zero

    def test_with_different_length_batches(self):
        # Test that different length batches raise an error
        with self.assertRaises(ValueError):
            batched_trajectory_distance_minimized(
                self.batch1, self.batch_different_size, euclidean_dist
            )

    def test_with_custom_distance_function(self):
        # Test with a custom distance function
        # noinspection PyUnusedLocal
        def custom_metric(traj1, traj2, **kwargs):
            return np.linalg.norm(traj1 - traj2)

        matching, minimized_distances, total_distance = batched_trajectory_distance_minimized(
            self.batch1, self.batch2, custom_metric
        )

        self.assertEqual(matching, [(0, 0), (1, 1)])  # Expecting a perfect match
        self.assertTrue(np.allclose(minimized_distances, [0.0, 0.0]))  # Should be zero distances for identical data
        self.assertAlmostEqual(total_distance, 0.0)

    def test_with_different_metrics(self):
        # Test with a different metric, e.g., sum of squared distances
        # noinspection PyUnusedLocal
        def squared_distance_metric(traj1, traj2, **kwargs):
            return np.sum((traj1 - traj2) ** 2)

        matching, minimized_distances, total_distance = batched_trajectory_distance_minimized(
            self.batch1, self.batch2, squared_distance_metric
        )

        self.assertEqual(matching, [(0, 0), (1, 1)])  # Expecting a perfect match
        self.assertTrue(np.allclose(minimized_distances, [0.0, 0.0]))  # Should be zero distances for identical data
        self.assertAlmostEqual(total_distance, 0.0)
