#!/usr/bin/env python3
"""Unittest for the metrics module."""
import logging
from typing import Any
from unittest import TestCase

import numpy as np
from haversine import haversine, Unit, haversine_vector
from scipy.spatial.distance import euclidean

from privtrajgen.metrics import *
# noinspection PyProtectedMember
from privtrajgen.metrics.distribution import sliced_wasserstein_distance, compute_heatmap, _JS_divergence, _JS_distance
from privtrajgen.metrics.trajectory_preservation import hausdorff_directed_haversine, euclidean_distance, \
    haversine_distance, reorder_dataset_to_matching
from privtrajgen.utils.helpers import printc


log = logging.getLogger(__name__)
def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


class TestDistributionMetrics(TestCase):

    def setUp(self):
        # Create two sets of trajectories for testing
        B, C, L = 10, 2, 100
        self.traj1 = np.random.randn(B, C, L)
        self.traj2 = np.random.randn(B, C, L)
        self.points1 = self.traj1.transpose(0, 2, 1).reshape(-1, C)
        self.points2 = self.traj2.transpose(0, 2, 1).reshape(-1, C)

    def test_JS_divergence(self):
        # compare two identical matrices
        matrix1 = np.random.rand(10, 10)
        self.assertEqual(
            _JS_divergence(matrix1, matrix1),
            0.0
        )
        # compare two different matrices
        matrix2 = np.random.rand(10, 10)
        self.assertLess(
            0.0,
            _JS_divergence(matrix1, matrix2)
        )

    def test_JS_distance(self):
        # compare two identical matrices
        matrix1 = np.random.rand(10, 10)
        self.assertEqual(
            _JS_distance(matrix1, matrix1),
            0.0
        )
        # compare two different matrices
        matrix2 = np.random.rand(10, 10)
        self.assertLess(
            0.0,
            _JS_distance(matrix1, matrix2)
        )

    def test_compute_heatmap(self):
        # Test simple heatmap creation
        points = np.array([[1.0, 1.0], [2.0, 2.0], [1.5, 1.5]])
        latgrids = np.linspace(0.0, 3.0, 4)
        longrids = np.linspace(0.0, 3.0, 4)
        heatmap = compute_heatmap(points, latgrids, longrids)

        # The heatmap should have a specific size
        self.assertEqual(heatmap.shape, (3, 3))
        # The heatmap should sum to 1 (since density=True)
        self.assertAlmostEqual(np.sum(heatmap), 1.0)

    def test_JS_from_points(self):
        # Test simple case with identical distributions
        bbox = {'lat_min': 0.0, 'lat_max': 3.0, 'lon_min': 0.0, 'lon_max': 3.0}
        points1 = np.array([[1.0, 1.0], [2.0, 2.0], [1.5, 1.5]])
        points2 = np.array([[1.0, 1.0], [2.0, 2.0], [1.5, 1.5]])
        grid_size = 3
        js_dist = JS_from_points(points1, points2, grid_size, bbox, mode='distance')
        self.assertAlmostEqual(js_dist, 0.0)

        # Test simple case with different distributions
        points2 = np.array([[0.5, 0.5], [2.5, 2.5], [1.0, 1.0]])
        js_dist = JS_from_points(points1, points2, grid_size, bbox, mode='distance')
        self.assertGreater(js_dist, 0.0)

        # Test divergence mode
        js_div = JS_from_points(points1, points2, grid_size, bbox, mode='divergence')
        self.assertGreater(js_div, 0.0)
        self.assertAlmostEqual(js_div, js_dist ** 2)

    def test_sliced_wasserstein_distance(self):
        # Test SWD between two identical distributions (SWD should be close to 0)
        samples1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        samples2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        swd = sliced_wasserstein_distance(samples1, samples2)
        self.assertAlmostEqual(swd, 0.0)

        # Test SWD between two different distributions
        samples2 = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
        swd = sliced_wasserstein_distance(samples1, samples2)
        self.assertGreater(swd, 0.0)

class TestHausdorffDistance(TestCase):

    def test_hausdorff_directed_haversine_identical(self):
        # Identical trajectories should have zero Hausdorff distance
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        direction1, direction2 = hausdorff_directed_haversine(t1, t2)
        self.assertAlmostEqual(direction1, 0.0)
        self.assertAlmostEqual(direction2, 0.0)

    def test_hausdorff_directed_haversine_different(self):
        # Test with different trajectories
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])

        direction1, direction2 = hausdorff_directed_haversine(t1, t2)
        self.assertGreater(direction1, 0.0)
        self.assertGreater(direction2, 0.0)

    def test_hausdorff_directed_haversine_distinct_points(self):
        # Test with distinct points in completely different locations
        t1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        t2 = np.array([[10.0, 10.0], [20.0, 20.0]])

        direction1, direction2 = hausdorff_directed_haversine(t1, t2)
        self.assertGreater(direction1, 0.0)
        self.assertGreater(direction2, 0.0)

    def test_hausdorff_distance_haversine(self):
        # Test with Haversine distance enabled
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])

        hd = hausdorff_distance(t1, t2, use_haversine=True)
        self.assertGreater(hd, 0.0)

    def test_hausdorff_distance_haversine_identical(self):
        # Identical trajectories should have zero Hausdorff distance
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        hd = hausdorff_distance(t1, t2, use_haversine=True)
        self.assertAlmostEqual(hd, 0.0)

    def test_hausdorff_distance_euclidean(self):
        # Test with Euclidean distance enabled
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])

        hd = hausdorff_distance(t1, t2, use_haversine=False)
        self.assertGreater(hd, 0.0)

    def test_hausdorff_distance_euclidean_identical(self):
        # Identical trajectories should have zero Hausdorff distance (Euclidean)
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        hd = hausdorff_distance(t1, t2, use_haversine=False)
        self.assertAlmostEqual(hd, 0.0)

    def test_empty_trajectories(self):
        # Edge case: Test with empty trajectories
        t1 = np.array([])
        t2 = np.array([])

        with self.assertRaises(ValueError):
            hausdorff_distance(t1, t2)

    def test_mismatched_batch_sizes(self):
        # Edge case: Test with mismatched trajectory sizes
        t1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        t2 = np.array([[0.0, 0.0]])

        with self.assertRaises(ValueError):
            hausdorff_distance(t1, t2)


class TestDTWDistance(TestCase):

    def test_dtw_distance_identical(self):
        # Identical trajectories should have zero DTW distance
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        dtw = dtw_distance(t1, t2)
        self.assertAlmostEqual(dtw, 0.0)

    def test_dtw_distance_different(self):
        # Test with different trajectories
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])

        dtw = dtw_distance(t1, t2)
        self.assertGreater(dtw, 0.0)

    def test_dtw_haversine(self):
        # Test with Haversine distance enabled
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])

        dtw = dtw_distance(t1, t1, use_haversine=True)
        self.assertAlmostEqual(dtw, 0.0)
        dtw = dtw_distance(t1, t2, use_haversine=True)
        self.assertGreater(dtw, 0.0)
        dtw_euclid = dtw_distance(t1, t2, use_haversine=False)
        self.assertNotEqual(
            dtw,
            dtw_euclid
        )
        printc("Euclidean DTW", dtw_euclid)
        printc("Haversine DTW", dtw)

    def test_empty_trajectories(self):
        # Edge case: Test with empty trajectories
        t1 = np.array([])
        t2 = np.array([])

        with self.assertRaises(ValueError):
            dtw_distance(t1, t2)

    def test_dtw_concrete(self):
        """Test the DTW distance with concrete examples."""
        t1 = np.array([[0.0, 0.0], [2.0, 2.0], [7.0, 7.0]])
        t2 = np.array([[0.0, 0.0], [3.0, 3.0], [4.0, 4.0]])
        dtw = dtw_distance(t1, t2)
        # This should be equal to the Euclidean distance between the two trajectories because for length 3
        # the start and end points need to be aligned.
        euclidean = np.sum(np.linalg.norm(t1 - t2, axis=1))
        self.assertAlmostEqual(dtw, euclidean)

    def test_dtw_concrete_haversine(self):
        """Test the DTW distance with concrete examples."""
        t1 = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
        t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])
        dtw = dtw_distance(t1, t2, use_haversine=True)
        # This should be equal to the Euclidean distance between the two trajectories because for length 3
        # the start and end points need to be aligned.
        # expected = np.sum(haversine_vector(t1, t2, unit=Unit.METERS))
        expected = np.sum([haversine(t1[i], t2[i], Unit.METERS) for i in range(len(t1))])
        self.assertAlmostEqual(dtw, expected)  # Differs by 23.949m

    def test_dtw_concrete_haversine_short(self):
        """Test the DTW distance with concrete examples."""
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [4.0, 4.0]])
        t2 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        dtw = dtw_distance(t1, t2, use_haversine=True)
        # This should be equal to the Euclidean distance between the two trajectories because for length 3
        # the start and end points need to be aligned.
        euclidean = np.sum(haversine_vector(t1, t2, unit=Unit.METERS))
        self.assertAlmostEqual(dtw, euclidean)

    def test_dtw_concrete_unequal_lengths(self):
        """Test the DTW distance with concrete examples."""
        t1 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        dtw = dtw_distance(t1, t2)
        matching_idx = { # t1->t2
            0: 0,
            1: 1,
            2: 2,
            3: 2
        }
        euclidean = np.sum([np.linalg.norm(t1[i] - t2[matching_idx[i]]) for i in range(len(t1))])
        self.assertAlmostEqual(dtw, euclidean)

    def test_dtw_concrete_unequal_lengths_haversine(self):
        """Test the DTW distance with concrete examples."""
        t1 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        dtw = dtw_distance(t1, t2, use_haversine=True)
        matching_idx = {0: 0, 1: 1, 2: 2, 3: 2}  # Assumed alignment
        expected_distance = sum([haversine(t1[i], t2[matching_idx[i]], Unit.METERS) for i in range(len(t1))])
        self.assertAlmostEqual(dtw, expected_distance)

    def test_compare_dtw_libs(self):
        """Compare the dtw-python and fastdtw implementations to make sure we use a correct computation."""
        from dtw import dtw
        from fastdtw import fastdtw
        t1 = np.random.randn(100, 2)
        t2 = np.random.randn(100, 2)

        # Compute the DTW distance using the fastdtw implementation
        distance, alignment = fastdtw(t1, t2, dist=euclidean, radius=max(len(t1), len(t2)))

        step_patterns = ["symmetric1", "symmetric2", "asymmetric"]
        for step_pattern in step_patterns:
            distance2 = dtw(t1, t2, distance_only=True, keep_internals=False,
                            dist_method="euclidean", step_pattern=step_pattern).distance
            logc(f"Match for {step_pattern}:", f"{distance2 - distance:.2f}")

        # fastdtw uses symmetric1 by default - that's also the simplest definition.
        expected_distance = dtw(t1, t2,distance_only=True, keep_internals=False,
                                step_pattern="symmetric1", dist_method="euclidean").distance
        self.assertAlmostEqual(distance, expected_distance)

class TestEuclideanDistance(TestCase):

        def test_euclidean_distance_identical(self):
            # Identical trajectories should have zero Euclidean distance
            t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

            euclidean_dist = euclidean_distance(t1, t2)
            self.assertAlmostEqual(euclidean_dist, 0.0)

        def test_euclidean_distance_different(self):
            # Test with different trajectories
            t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])

            euclidean_dist = euclidean_distance(t1, t2)
            expected_dist = np.linalg.norm([1.0, 1.0]) * 2
            self.assertAlmostEqual(euclidean_dist, expected_dist)

        def test_euclidean_distance_empty(self):
            # Test with empty trajectories
            t1 = np.array([])
            t2 = np.array([])

            with self.assertRaises(ValueError):
                euclidean_distance(t1, t2)

        def test_euclidean_distance_unequal_lengths(self):
            # Test with trajectories of unequal lengths
            t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            t2 = np.array([[0.0, 0.0], [1.0, 1.0]])

            with self.assertRaises(ValueError):
                euclidean_distance(t1, t2)

        def test_euclidean_vs_dtw(self):
            """For two trajectories with i:i mapping, the Euclidean and dtw distance should be equal.
            Only true if Euclidean distance is defined as np.sum(np.linalg.norm(t1 - t2, axis=1))
            and not np.linalg.norm(t1 - t2)
            """
            t1 = np.array([[0.0, 0.0], [5.0, 5.0], [20.0, 20.0]])
            t2 = np.array([[1.0, 1.0], [10.0, 10.0], [25.0, 25.0]])
            euclidean_dist = euclidean_distance(t1, t2, normalise=False)
            dtw_dist, alignment = dtw_distance(t1, t2, return_alignment=True)
            # print(alignment)
            self.assertAlmostEqual(euclidean_dist, dtw_dist)

class TestHaversineDistance(TestCase):

    def test_haversine_distance_identical(self):
        # Identical trajectories should have zero haversine distance
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        haversine_dist = haversine_distance(t1, t2)
        self.assertAlmostEqual(haversine_dist, 0.0)

    def test_haversine_distance_different(self):
        # Test with different trajectories
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])

        haversine_dist = haversine_distance(t1, t2)
        expected_dist = haversine_vector(t1, t2, Unit.METERS).sum()
        self.assertAlmostEqual(haversine_dist, expected_dist)

    def test_haversine_distance_empty(self):
        # Test with empty trajectories
        t1 = np.array([])
        t2 = np.array([])

        with self.assertRaises(ValueError):
            haversine_distance(t1, t2)

    def test_haversine_distance_unequal_lengths(self):
        # Test with trajectories of unequal lengths
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0]])

        with self.assertRaises(ValueError):
            haversine_distance(t1, t2)

    def test_haversine_vs_dtw(self):
        """For two trajectories with i:i mapping, the haversine and dtw distance should be equal.
        Only true if haversine distance is defined as haversine_vector(t1, t2, Unit.METERS).sum()
        """
        t1 = np.array([[0.0, 0.0], [5.0, 5.0], [20.0, 20.0]])
        t2 = np.array([[1.0, 1.0], [10.0, 10.0], [25.0, 25.0]])
        haversine_dist = haversine_distance(t1, t2, normalise=False)
        dtw_dist, alignment = dtw_distance(t1, t2, return_alignment=True, use_haversine=True)
        # print(alignment)
        self.assertAlmostEqual(haversine_dist, dtw_dist)

class TestNormalisedEuclideanDistance(TestCase):

    def test_normalised_euclidean_distance_identical(self):
        # Identical trajectories should have zero normalised Euclidean distance
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        normalised_euclidean_dist = normalised_euclidean_distance(t1, t2)
        self.assertAlmostEqual(normalised_euclidean_dist, 0.0)

    def test_normalised_euclidean_distance_different(self):
        # Test with different trajectories
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])

        normalised_euclidean_dist = normalised_euclidean_distance(t1, t2)
        expected_dist = np.linalg.norm([1.0, 1.0]) * 2 / len(t1)
        self.assertAlmostEqual(normalised_euclidean_dist, expected_dist)

    def test_normalised_euclidean_distance_empty(self):
        # Test with empty trajectories
        t1 = np.array([])
        t2 = np.array([])

        with self.assertRaises(ValueError):
            normalised_euclidean_distance(t1, t2)

    def test_normalised_euclidean_distance_unequal_lengths(self):
        # Test with trajectories of unequal lengths
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0]])

        with self.assertRaises(ValueError):
            normalised_euclidean_distance(t1, t2)


class TestNormalisedHaversineDistance(TestCase):

    def test_normalised_haversine_distance_identical(self):
        # Identical trajectories should have zero normalised haversine distance
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        normalised_haversine_dist = normalised_haversine_distance(t1, t2)
        self.assertAlmostEqual(normalised_haversine_dist, 0.0)

    def test_normalised_haversine_distance_different(self):
        # Test with different trajectories
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])

        normalised_haversine_dist = normalised_haversine_distance(t1, t2)
        expected_dist = haversine_vector(t1, t2, Unit.METERS).sum() / len(t1)
        self.assertAlmostEqual(normalised_haversine_dist, expected_dist)

    def test_normalised_haversine_distance_empty(self):
        # Test with empty trajectories
        t1 = np.array([])
        t2 = np.array([])

        with self.assertRaises(ValueError):
            normalised_haversine_distance(t1, t2)

    def test_normalised_haversine_distance_unequal_lengths(self):
        # Test with trajectories of unequal lengths
        t1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        t2 = np.array([[0.0, 0.0], [1.0, 1.0]])

        with self.assertRaises(ValueError):
            normalised_haversine_distance(t1, t2)


class TestReorderDatasetToMatching(TestCase):

    def setUp(self):
        # Set up a simple dataset and matching for testing
        self.dataset = [np.array([i]) for i in range(5)]  # Dataset with 5 elements: [0], [1], [2], [3], [4]
        self.matching = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]  # Match indices in reverse order

    def test_reorder_dataset(self):
        # Expected reordered dataset is [4], [3], [2], [1], [0]
        expected = [np.array([4]), np.array([3]), np.array([2]), np.array([1]), np.array([0])]
        result = reorder_dataset_to_matching(self.dataset, self.matching)

        for res, exp in zip(result, expected):
            np.testing.assert_array_equal(res, exp)

    def test_empty_dataset(self):
        # Test with an empty dataset
        dataset = []
        matching = []
        expected = []
        result = reorder_dataset_to_matching(dataset, matching)
        self.assertEqual(result, expected)

    def test_shorter(self):
        # Test with partial reordering
        dataset = [np.array([i]) for i in range(5)]
        matching = [(0, 1), (1, 0)]  # Only reorder the first two elements
        with self.assertRaises(ValueError):
            reorder_dataset_to_matching(dataset, matching)