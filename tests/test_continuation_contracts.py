from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ContinuationContractTest(unittest.TestCase):
    def test_description_priority_requires_every_completed_run(self):
        text = (ROOT / 'scripts' / 'continue_hits10_description_after_wave2.sh').read_text()
        self.assertEqual(text.count('--minimum-complete 8'), 1)
        self.assertEqual(text.count('--minimum-complete 12'), 1)
        self.assertEqual(text.count('--required-run-manifest'), 5)
        self.assertIn('GPU_IDS=0,1,2,3', text)

    def test_sequential_wave5_cannot_ignore_missing_branches(self):
        text = (ROOT / 'scripts' / 'continue_hits10_after_wave4.sh').read_text()
        self.assertEqual(text.count('--minimum-complete 16'), 1)
        self.assertEqual(text.count('--minimum-complete 20'), 1)
        self.assertEqual(text.count('--required-run-manifest'), 9)

    def test_description_wave_is_a_late_schedule_grid(self):
        text = (ROOT / 'scripts' / 'launch_zh_hits10_description_scale.sh').read_text()
        self.assertNotIn('2.50 constant 2.50', text)
        self.assertEqual(text.count('1.00 step-increase 2.50 150'), 1)
        self.assertIn('1.00 step-increase 2.50 151', text)
        self.assertIn('1.00 step-increase 2.55 151', text)
        self.assertIn('1.00 step-increase 2.65 151', text)
        self.assertNotIn('"$evidence_lr" constant', text)
        self.assertNotIn('"$evidence_lr" single-step 150 0.10', text)
        self.assertNotIn('"$evidence_lr" single-step 151 0.000001\n', text)
        self.assertEqual(text.count('"$evidence_lr" single-step 151 0.000000001'), 3)
        self.assertIn('1to2p55_step151_lr2p5_nearfreeze', text)
        self.assertIn('1to2p65_step151_lr2p5_nearfreeze', text)
        self.assertIn('EVIDENCE_LR:-2.5e-6', text)
        self.assertIn('--snapshot_epochs 75,100,150,151,200,250,299', text)
        self.assertEqual(text.count(' evidence\n'), 3)
        self.assertIn('config_family', text)
        self.assertIn('run_temperature=0.08', text)
        self.assertIn('run_temperature_schedule=constant', text)
        self.assertIn('run_reverse=0.5', text)
        self.assertIn('run_reverse_schedule=constant', text)
        self.assertIn('run_beta=0.9', text)
        self.assertIn('run_source_inbatch=pseudo-target', text)
        self.assertIn('run_fnmask=0', text)

    def test_epoch151_monitor_is_cpu_only_and_full_target(self):
        text = (ROOT / 'scripts' / 'monitor_description_epoch151.sh').read_text()
        self.assertIn('snapshot_epoch_151.pt', text)
        self.assertIn('--report-epoch 151', text)
        self.assertIn('nearfreeze_count" -ne 3', text)
        self.assertIn('for nearfreeze_checkpoint in', text)
        self.assertIn('CUDA_VISIBLE_DEVICES=""', text)
        self.assertIn('renice 19 -p "$$"', text)
        self.assertIn('taskset -c 80-83', text)
        self.assertIn('scripts/audit_checkpoint_drift.py', text)
        self.assertIn('--maximum-absolute-delta 1e-10', text)
        self.assertIn('--require-before-epoch 150', text)
        self.assertIn('--require-after-epoch 151', text)
        self.assertIn('--candidate-scope full-target', text)
        self.assertIn('--weight-source online', text)
        self.assertIn('--device cpu', text)


if __name__ == '__main__':
    unittest.main()
