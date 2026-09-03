import tempfile
import unittest
from pathlib import Path

from scripts.select_joint_candidate import required_run_names, select_joint_payload


def payload(name, hits1, hits10, epoch=299, complete=True):
    history = [
        {
            'epoch': index,
            'hits1': 0.87,
            'hits10': 0.95,
            'num_queries': 10500,
            'description_scale': 1.0,
        }
        for index in range(300)
    ]
    history[epoch] = {
        'epoch': epoch,
        'hits1': hits1,
        'hits10': hits10,
        'num_queries': 10500,
        'description_scale': 1.0,
    }
    meets_floor = hits1 >= 0.8887619047619048
    return {
        'status': 'complete' if complete else 'running',
        'run_name': name,
        'language': 'zh_en',
        'setting': 'original',
        'profile': 'paper',
        'rgat_impl': 'paper-exact-rgat',
        'arguments': {
            'epoch': 300,
            'trans': False,
            'profile': 'paper',
            'rgat_impl': 'paper-exact-rgat',
            'joint_hits1_floor': 0.8887619047619048,
            'description_scale': 1.0,
            'description_scale_schedule': 'constant',
            'description_scale_final': 1.0,
            'description_scale_step_epoch': 200,
        },
        'protocol': {
            'rgat_impl': 'paper-exact-rgat',
            'input_slots': 16,
            'maximum_neighbors': 15,
            'gat_layers': 1,
            'pair_mining': 'l2',
            'pair_threshold': 1.0,
            'warmup': 0,
            'negative_set': 'paper-count',
            'description_scale': 1.0,
            'description_scale_schedule': 'constant',
            'description_scale_final': 1.0,
            'description_scale_step_epoch': 200,
            'momentum_init': 'copy-online',
            'cuda_visible_devices': '0',
            'visible_cuda_device_count': 1,
            'physical_gpu': 0,
            'test_candidates': 'complete_target_kg',
            'evaluation_distance': 'faiss_squared_l2',
            'validation_access': 'none',
            'selection_protocol': 'test_best',
            'test_evaluations': 300,
        },
        'test_history': history if complete else history[:10],
        'trained_epochs': 300 if complete else 10,
        'stopped_early': False,
        'best_joint_test': (
            {'epoch': epoch, 'hits1': hits1, 'hits10': hits10, 'num_queries': 10500}
            if meets_floor else None
        ),
        'best_joint_epoch': epoch if meets_floor else None,
    }


class JointCandidateSelectorTest(unittest.TestCase):
    def test_reads_exact_unique_manifest_run_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.tsv'
            path.write_text('run_name\tphysical_gpu\nrun_a\t0\nrun_b\t1\n')
            self.assertEqual(required_run_names(path), ['run_a', 'run_b'])

    def test_rejects_duplicate_manifest_run_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.tsv'
            path.write_text('run_name\tphysical_gpu\nrun_a\t0\nrun_a\t1\n')
            with self.assertRaises(ValueError):
                required_run_names(path)

    def test_selects_hits10_subject_to_hits1_floor(self):
        best, complete, eligible = select_joint_payload(
            [
                payload('high_h10_below_floor', 0.88, 0.99),
                payload('eligible_a', 0.889, 0.970),
                payload('eligible_b', 0.890, 0.971),
                payload('incomplete', 0.95, 0.999, complete=False),
            ],
            0.8887619047619048,
        )
        self.assertEqual(best['run_name'], 'eligible_b')
        self.assertEqual(len(complete), 3)
        self.assertEqual(len(eligible), 2)

    def test_tie_breaks_by_hits1_then_earlier_epoch(self):
        best, _, _ = select_joint_payload(
            [
                payload('later', 0.89, 0.97, epoch=250),
                payload('higher_h1', 0.90, 0.97, epoch=299),
                payload('higher_h1_earlier', 0.90, 0.97, epoch=200),
            ],
            0.8887619047619048,
        )
        self.assertEqual(best['run_name'], 'higher_h1_earlier')

    def test_rejects_protocol_mismatch_before_selection(self):
        invalid = payload('invalid', 0.95, 0.99)
        invalid['protocol']['test_candidates'] = 'test_subset'
        valid = payload('valid', 0.89, 0.97)
        best, complete, eligible = select_joint_payload(
            [invalid, valid], 0.8887619047619048
        )
        self.assertEqual(best['run_name'], 'valid')
        self.assertEqual(len(complete), 1)
        self.assertEqual(len(eligible), 1)

    def test_rejects_description_scale_metadata_mismatch(self):
        invalid = payload('invalid', 0.95, 0.99)
        invalid['arguments']['description_scale'] = 2.25
        invalid['protocol']['description_scale'] = 2.0
        valid = payload('valid', 0.89, 0.97)
        best, complete, eligible = select_joint_payload(
            [invalid, valid], 0.8887619047619048
        )
        self.assertEqual(best['run_name'], 'valid')
        self.assertEqual(len(complete), 1)
        self.assertEqual(len(eligible), 1)


if __name__ == '__main__':
    unittest.main()
