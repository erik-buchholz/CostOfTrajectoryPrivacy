#!/usr/bin/env python3
"""Unittest for computation of conditional attributes for trajectories."""
from unittest import TestCase

from privtrajgen.datasets.conditional_info import *
from privtrajgen.config.constants import *
from privtrajgen.utils.grid import compute_grid_cell


class TestConditionalAttributes(TestCase):
    def test_compute_conditional_info(self):
        # Define sample data and bbox
        data = {LAT: [35, 36], LON: [139, 140], TIMESTAMP: [pd.Timestamp('2022-01-01 10:00'), pd.Timestamp('2022-01-01 11:00')]}
        bbox = {'lat_min': 34, 'lat_max': 37, 'lon_min': 138, 'lon_max': 141}
        traj = pd.DataFrame(data)
        result = compute_conditional_info(traj, bbox)
        # Test the output structure and some example checks
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 8)
        # Retry as dict
        result = compute_conditional_info(traj, bbox, as_dict=True)
        self.assertIsInstance(result, dict)
        self.assertTrue("dep_time" in result)
        self.assertTrue("total_dist" in result)
        self.assertTrue("total_dur" in result)
        self.assertTrue("len" in result)
        self.assertTrue("avg_speed" in result)
        self.assertTrue("sid" in result)
        self.assertTrue("eid" in result)

    def test_compute_departure_time(self):
        data = {HOUR: [10], DOW: [5], TIMESTAMP: [pd.Timestamp('2022-01-01 10:30')]}  # Saturday 10:30
        traj = pd.DataFrame(data)
        dep_time = compute_departure_time(traj)
        self.assertEqual(dep_time, 126)
        dep_time = compute_departure_time(traj, categorical_timestamp=True)
        self.assertEqual(dep_time, 120)  # Only hour is considered

    def test_compute_departure_time_as_category(self):
        hour = 14
        minute = 22
        category = compute_departure_time_as_category(hour, minute)
        self.assertEqual(category, 172)

    def test_compute_total_and_avg_distance(self):
        lat_lon = np.array([[35, 139], [36, 140]])
        total_distance, avg_distance = compute_total_and_avg_distance(lat_lon)
        self.assertGreater(total_distance, 0)
        self.assertAlmostEqual(total_distance, avg_distance)  # Since only one distance is computed

    def test_compute_total_duration(self):
        data = {TIMESTAMP: [pd.Timestamp('2022-01-01 10:00'), pd.Timestamp('2022-01-01 11:00')]}
        traj = pd.DataFrame(data)
        duration = compute_total_duration(traj)
        self.assertEqual(duration, 3600)

    def test_compute_total_length(self):
        data = {LAT: [35, 36], LON: [139, 140]}
        traj = pd.DataFrame(data)
        length = compute_total_length(traj)
        self.assertEqual(length, 2)

    def test_compute_avg_speed(self):
        avg_speed = compute_avg_speed(1000, 100)
        self.assertEqual(avg_speed, 10.0)

    def test_compute_grid_cell(self):
        latgrids = np.array([34, 35, 36, 37, 38])
        longrids = np.array([138, 139, 140, 141, 142])
        grid_id = compute_grid_cell(lat=35.5, lon=139.5, latgrids=latgrids, longrids=longrids)
        self.assertEqual(grid_id, 5)
        # Out of bounds should raise ValueError
        with self.assertRaises(ValueError):
            # Lat too low
            compute_grid_cell(lat=30, lon=140, latgrids=latgrids, longrids=longrids)
            # Lat too high
            compute_grid_cell(lat=39, lon=140, latgrids=latgrids, longrids=longrids)
            # Lon too low
            compute_grid_cell(lat=35, lon=137, latgrids=latgrids, longrids=longrids)
            # Lon too high
            compute_grid_cell(lat=35, lon=143, latgrids=latgrids, longrids=longrids)
        # Try the smallest possible cell id
        grid_id = compute_grid_cell(lat=34, lon=138, latgrids=latgrids, longrids=longrids)
        self.assertEqual(grid_id, 0)
        # Try the largest possible cell id
        grid_id = compute_grid_cell(lat=38, lon=142, latgrids=latgrids, longrids=longrids)
        self.assertEqual(grid_id, (len(latgrids) - 1) * (len(longrids) - 1) - 1)


