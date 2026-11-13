"""Offline tests; no ROS installation or physical robot required."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from amr_ab_control.approach_core import (A, B, MapGuard, standoff_goal,
                           find_approach_goal, eligible_target_class, robust_depth_m)


class LogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = MapGuard(ROOT / 'maps' / 'minimap.yaml')

    def test_standoff_distance(self):
        x, y, heading = standoff_goal((0.0, 0.0), (2.0, 0.0), 0.8)
        self.assertAlmostEqual(x, 1.2)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(heading, 0.0)

    def test_already_close(self):
        self.assertIsNone(standoff_goal((0,0), (0.5,0), 0.8))

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            standoff_goal((0,0), (math.nan,0))

    def test_only_car_allowed(self):
        self.assertTrue(eligible_target_class('car'))
        self.assertFalse(eligible_target_class('dummy'))
        self.assertFalse(eligible_target_class('person'))

    def test_depth_with_holes_mm(self):
        depth = np.full((30,30), 1200, dtype='uint16')
        depth[14:16,14:16] = 0
        z = robust_depth_m(depth, (10,10,20,20), '16UC1')
        self.assertAlmostEqual(z, 1.2)

    def test_depth_in_meters(self):
        depth = np.full((30,30), 1.5, dtype='float32')
        self.assertAlmostEqual(robust_depth_m(depth, (10,10,20,20), '32FC1'), 1.5)

    def test_depth_invalid(self):
        depth = np.zeros((20,20), dtype='uint16')
        self.assertIsNone(robust_depth_m(depth,(4,4,16,16),'16UC1'))

    def test_b_c_navigable(self):
        self.assertTrue(self.guard.safe_goal(A))
        self.assertTrue(self.guard.safe_goal(B))

    def test_other_zone_rejected(self):
        self.assertFalse(self.guard.safe_goal((-2.5,-7.3)))
        self.assertFalse(self.guard.target_in_zone((2,20)))

    def test_approach_goal_inside_corridor(self):
        goal = find_approach_goal(A, (-1.1,-0.2), self.guard, 0.8)
        self.assertTrue(self.guard.safe_goal(goal[:2]))

    def test_target_outside_corridor(self):
        with self.assertRaises(ValueError):
            find_approach_goal(A, (-2.5,-7.3), self.guard, 0.8)


if __name__ == '__main__':
    unittest.main()
