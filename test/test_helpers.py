#!/usr/bin/env python3
""" """
from unittest import TestCase
from timeit import default_timer as timer
from datetime import datetime

import numpy as np

from privtrajgen.utils import helpers, mean_confidence_interval


class Test(TestCase):
    def test_compute_inverse_power(self):
        self.assertEqual(
            5,
            helpers.compute_inverse_power(1 / 60000)
        )
        self.assertEqual(
            8,
            helpers.compute_inverse_power(1 / 100000000)
        )

    def test_compute_delta(self):
        self.assertLessEqual(
            helpers.compute_delta(50000),
            1 / 50000 ** 1.1
        )
        self.assertLessEqual(
            helpers.compute_delta(55555),
            1 / 50000 ** 1.1
        )
        self.assertLessEqual(
            helpers.compute_delta(100000),
            1 / 100000 ** 1.1,
        )
        self.assertLessEqual(
            helpers.compute_delta(60000) ** 1.1,
            1e-5
        )

    def test_time_to_str(self):
        start_timeit = timer()
        start_datetime = datetime.now()
        self.assertEqual(
            "01:00:00h",
            helpers.time_to_str(3600)
        )
        self.assertEqual(
            "01:00min",
            helpers.time_to_str(60)
        )
        self.assertEqual(
            "1.00s",
            helpers.time_to_str(1)
        )

        # Test compatibility with default_timer
        # Verify format via regex
        self.assertRegex(
            helpers.time_to_str(timer() - start_timeit),
            r"\d{1}.\d{2}s"
        )

        # Test compatibility with datetime
        # Verify format via regex
        self.assertRegex(
            helpers.time_to_str(datetime.now() - start_datetime),
            r"\d{1}.\d{2}s"
        )

class TestMeanConfidenceInterval(TestCase):
    def setUp(self):
        """Set up sample data for testing."""
        self.data = [2.3, 2.5, 2.8, 3.0, 3.1, 3.3, 3.5]
        self.confidence = 0.95

    def test_mean_confidence_interval(self):
        """Test mean_confidence_interval function with valid data."""
        mean, lower_bound, upper_bound = mean_confidence_interval(self.data, self.confidence)
        self.assertAlmostEqual(mean, np.mean(self.data), places=5)
        self.assertTrue(lower_bound < mean < upper_bound)

    def test_empty_data(self):
        """Test mean_confidence_interval with empty data."""
        with self.assertRaises(ValueError):
            mean_confidence_interval([], self.confidence)

    def test_invalid_confidence(self):
        """Test mean_confidence_interval with invalid confidence value."""
        with self.assertRaises(ValueError):
            mean_confidence_interval(self.data, 1.5)
        with self.assertRaises(ValueError):
            mean_confidence_interval(self.data, -0.1)

    def test_numpy_array_input(self):
        """Test mean_confidence_interval with NumPy array input."""
        numpy_data = np.array(self.data)
        mean, lower_bound, upper_bound = mean_confidence_interval(numpy_data, self.confidence)
        self.assertAlmostEqual(mean, np.mean(numpy_data), places=5)
        self.assertTrue(lower_bound < mean < upper_bound)

    def test_concrete_values(self):
        """Test mean_confidence_interval with concrete expected values."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        confidence = 0.95
        mean, lower_bound, upper_bound = mean_confidence_interval(data, confidence)
        self.assertAlmostEqual(mean, 3.0, places=5)
        self.assertAlmostEqual(lower_bound, 1.036756838522439, places=5)
        self.assertAlmostEqual(upper_bound, 4.963243161477561, places=5)




