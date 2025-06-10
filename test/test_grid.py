#!/usr/bin/env python3
"""Test Grid and Grid Cell computations."""
import unittest
import numpy as np

from privtrajgen.utils.grid import divide_grids, compute_grid_cell


class TestGridFunctions(unittest.TestCase):

    def test_divide_grids(self):
        # Test case 1: Typical case
        boundary = {'lat_min': 0.0, 'lat_max': 12.0, 'lon_min': 0.0, 'lon_max': 12.0}
        latgrids, longrids = divide_grids(boundary, grids_num=12)
        expected_latgrids = np.linspace(0.0, 12.0, 13)
        expected_longrids = np.linspace(0.0, 12.0, 13)
        np.testing.assert_array_almost_equal(latgrids, expected_latgrids)
        np.testing.assert_array_almost_equal(longrids, expected_longrids)

        # Test case 2: Non-square grid
        boundary = {'lat_min': 0.0, 'lat_max': 10.0, 'lon_min': 0.0, 'lon_max': 20.0}
        latgrids, longrids = divide_grids(boundary, grids_num=10)
        expected_latgrids = np.linspace(0.0, 10.0, 11)
        expected_longrids = np.linspace(0.0, 20.0, 11)
        np.testing.assert_array_almost_equal(latgrids, expected_latgrids)
        np.testing.assert_array_almost_equal(longrids, expected_longrids)

        # Test case 3: Small grid with decimals
        boundary = {'lat_min': 1.5, 'lat_max': 3.5, 'lon_min': 2.0, 'lon_max': 5.0}
        latgrids, longrids = divide_grids(boundary, grids_num=2)
        expected_latgrids = np.linspace(1.5, 3.5, 3)
        expected_longrids = np.linspace(2.0, 5.0, 3)
        np.testing.assert_array_almost_equal(latgrids, expected_latgrids)
        np.testing.assert_array_almost_equal(longrids, expected_longrids)

    def test_compute_grid_cell(self):
        # Test case 1: Standard grid
        boundary = {'lat_min': 0.0, 'lat_max': 12.0, 'lon_min': 0.0, 'lon_max': 12.0}
        latgrids, longrids = divide_grids(boundary, grids_num=12)

        # Check points at exact grid center
        self.assertEqual(compute_grid_cell(6.0, 6.0, longrids, latgrids), 78)

        # Test case 2: Check boundaries
        self.assertEqual(compute_grid_cell(0.0, 0.0, longrids, latgrids), 0)  # Bottom-left corner
        self.assertEqual(compute_grid_cell(12.0, 12.0, longrids, latgrids), 143)  # Top-right corner

        # Test case 3: Check points at borders but within the grid
        self.assertEqual(compute_grid_cell(12.0, 6.0, longrids, latgrids), 83)
        self.assertEqual(compute_grid_cell(6.0, 12.0, longrids, latgrids), 138)

        # Test case 4: Point outside the grid
        with self.assertRaises(ValueError):
            compute_grid_cell(13.0, 13.0, longrids, latgrids)

        # Test case 5: Non-square grid
        boundary = {'lat_min': 0.0, 'lat_max': 10.0, 'lon_min': 0.0, 'lon_max': 20.0}
        latgrids, longrids = divide_grids(boundary, grids_num=10)

        # Check points near the borders
        self.assertEqual(compute_grid_cell(20.0, 10.0, longrids, latgrids), 99)
        self.assertEqual(compute_grid_cell(0.0, 0.0, longrids, latgrids), 0)

    def test_edge_cases(self):
        # Test case 1: Small grid of size 1x1
        boundary = {'lat_min': 0.0, 'lat_max': 1.0, 'lon_min': 0.0, 'lon_max': 1.0}
        latgrids, longrids = divide_grids(boundary, grids_num=1)
        self.assertEqual(compute_grid_cell(0.5, 0.5, longrids, latgrids), 0)
        self.assertEqual(compute_grid_cell(1.0, 1.0, longrids, latgrids), 0)  # Max boundary included

        # Test case 2: Negative boundaries
        boundary = {'lat_min': -10.0, 'lat_max': 0.0, 'lon_min': -20.0, 'lon_max': -10.0}
        latgrids, longrids = divide_grids(boundary, grids_num=10)
        self.assertEqual(compute_grid_cell(-15.0, -5.0, longrids, latgrids), 55)
        self.assertEqual(compute_grid_cell(-20.0, -10.0, longrids, latgrids), 0)


if __name__ == '__main__':
    unittest.main()

