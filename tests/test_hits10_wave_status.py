import tempfile
import unittest
from pathlib import Path

from scripts.status_hits10_wave import metric_at_epoch, read_metrics, read_pseudo_stats


class Hits10WaveStatusTest(unittest.TestCase):
    def test_parses_metrics_and_collision_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.log'
            path.write_text(
                'Test: epoch=7 #Entity=10500 Hits@1=0.8348 Hits@10=0.9398\n'
                'Pseudo pairs: 19022 + 19112 unique_targets=16087+16211 '
                'collisions=2935+2901 (mining=l2 threshold=1.0)\n'
            )
            self.assertEqual(read_metrics(str(path)), [(7, 0.8348, 0.9398)])
            stats = read_pseudo_stats(str(path))
            self.assertEqual(stats[0]['pairs'], [19022, 19112])
            self.assertEqual(stats[0]['collisions'], [2935, 2901])
            self.assertAlmostEqual(stats[0]['collision_rate'], 5836.0 / 38134.0)

    def test_selects_only_the_exact_requested_epoch(self):
        metrics = [(74, 0.87, 0.96), (75, 0.88, 0.97), (76, 0.89, 0.98)]
        self.assertEqual(metric_at_epoch(metrics, 75), (75, 0.88, 0.97))
        self.assertIsNone(metric_at_epoch(metrics, 77))


if __name__ == '__main__':
    unittest.main()
