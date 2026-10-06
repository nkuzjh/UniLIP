import unittest

import torch

from unilip.angle_utils import shortest_angle_distance


class ShortestAngleDistanceTests(unittest.TestCase):
    def test_float_wrap_and_multiple_turns(self):
        self.assertAlmostEqual(shortest_angle_distance(358.0, 360.0), 2.0)
        self.assertAlmostEqual(shortest_angle_distance(-358.0, 360.0), 2.0)
        self.assertAlmostEqual(shortest_angle_distance(720.0, 360.0), 0.0)
        self.assertAlmostEqual(shortest_angle_distance(1081.0, 360.0), 1.0)
        self.assertAlmostEqual(shortest_angle_distance(180.0, 360.0), 180.0)
        self.assertAlmostEqual(shortest_angle_distance(0.98, 1.0), 0.02)

    def test_tensor_wrap_nonnegative_for_out_of_range_predictions(self):
        difference = torch.tensor([358.0, -358.0, 720.0, 1081.0, 180.0])
        actual = shortest_angle_distance(difference, 360.0)
        torch.testing.assert_close(actual, torch.tensor([2.0, 2.0, 0.0, 1.0, 180.0]))
        self.assertTrue(bool(torch.all(actual >= 0)))
        normalized = shortest_angle_distance(torch.tensor([0.98, 2.25, -3.25]), 1.0)
        torch.testing.assert_close(normalized, torch.tensor([0.02, 0.25, 0.25]))


if __name__ == "__main__":
    unittest.main()
