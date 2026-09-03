import tempfile
import unittest
from pathlib import Path

import torch

from scripts.audit_checkpoint_drift import compare_checkpoints


class CheckpointDriftAuditTest(unittest.TestCase):
    def write_checkpoint(self, root, name, epoch, values):
        path = root / name
        torch.save({
            'epoch': epoch,
            'online_model': {
                'weight': torch.tensor(values, dtype=torch.float32),
                'counter': torch.tensor([3], dtype=torch.int64),
            },
        }, str(path))
        return path

    def test_reports_exact_float_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = self.write_checkpoint(root, 'before.pt', 150, [1.0, 2.0])
            after = self.write_checkpoint(root, 'after.pt', 151, [1.0, 2.25])
            report = compare_checkpoints(before, after, 'online')
        self.assertEqual(report['before_epoch'], 150)
        self.assertEqual(report['after_epoch'], 151)
        self.assertEqual(report['changed_tensor_count'], 1)
        self.assertEqual(report['changed_element_count'], 1)
        self.assertAlmostEqual(report['maximum_absolute_delta'], 0.25)

    def test_rejects_state_dict_key_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = self.write_checkpoint(root, 'before.pt', 150, [1.0])
            after = root / 'after.pt'
            torch.save({
                'epoch': 151,
                'online_model': {'other': torch.tensor([1.0])},
            }, str(after))
            with self.assertRaisesRegex(ValueError, 'key mismatch'):
                compare_checkpoints(before, after, 'online')


if __name__ == '__main__':
    unittest.main()
