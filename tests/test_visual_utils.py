import unittest

from visual_utils import _bounded_visualization_sample_count


class VisualizationSampleCountTests(unittest.TestCase):
    def test_caps_requested_samples_to_dataset_length(self):
        self.assertEqual(_bounded_visualization_sample_count(range(10), 20), 10)

    def test_empty_dataset_returns_zero(self):
        self.assertEqual(_bounded_visualization_sample_count([], 20), 0)

    def test_rejects_non_positive_or_non_integer_counts(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _bounded_visualization_sample_count(range(10), value)


if __name__ == "__main__":
    unittest.main()
