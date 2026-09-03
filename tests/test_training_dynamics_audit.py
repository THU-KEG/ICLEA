import tempfile
import unittest
from pathlib import Path

from scripts.audit_training_dynamics import parse_log, pearson, window_summary


class TrainingDynamicsAuditTest(unittest.TestCase):
    def test_aligns_epoch_loss_pseudo_pairs_and_metrics(self):
        text = """Epoch 0 learning_rate=2e-06 temperature=0.08
Pseudo pairs: 10 + 20 unique_targets=8+17 collisions=2+3
epoch=0 batch=10 step=200 loss=0.5
epoch=0 batch=20 step=400 loss=0.3
Test: epoch=0 queries=2 Hits@1=0.5 Hits@10=1.0
Epoch 1 learning_rate=1e-06 temperature=0.08
Pseudo pairs: 10 + 20 unique_targets=9+18 collisions=1+2
epoch=1 batch=10 step=600 loss=0.2
Test: epoch=1 queries=2 Hits@1=1.0 Hits@10=1.0
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.log'
            path.write_text(text)
            rows = parse_log(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['loss_sample_count'], 2)
        self.assertAlmostEqual(rows[0]['loss_sample_mean'], 0.4)
        self.assertAlmostEqual(rows[0]['pseudo_collision_rate'], 5.0 / 30.0)
        self.assertEqual(rows[1]['hits1'], 1.0)

    def test_pearson_and_windows_are_descriptive(self):
        rows = [
            {'epoch': 0, 'hits1': 0.1, 'hits10': 0.2, 'loss_sample_mean': 3.0},
            {'epoch': 1, 'hits1': 0.2, 'hits10': 0.3, 'loss_sample_mean': 2.0},
            {'epoch': 2, 'hits1': 0.3, 'hits10': 0.4, 'loss_sample_mean': 1.0},
        ]
        self.assertAlmostEqual(pearson(rows, 'loss_sample_mean', 'hits10'), -1.0)
        windows = window_summary(rows, size=2)
        self.assertEqual(len(windows), 2)
        self.assertAlmostEqual(windows[0]['mean_hits10'], 0.25)

    def test_rejects_unscoped_pseudo_pair_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.log'
            path.write_text(
                'Pseudo pairs: 1 + 1 unique_targets=1+1 collisions=0+0\n'
            )
            with self.assertRaisesRegex(ValueError, 'before an epoch'):
                parse_log(path)


if __name__ == '__main__':
    unittest.main()
