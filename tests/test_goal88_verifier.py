import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_goal88 import audit


class Goal88VerifierTest(unittest.TestCase):
    def payload(self):
        history = [
            {'epoch': epoch, 'hits1': 0.884, 'hits10': 0.97, 'num_queries': 10500}
            for epoch in range(300)
        ]
        return {
            'status': 'complete',
            'run_name': 'strict-pass',
            'language': 'zh_en',
            'setting': 'original',
            'arguments': {'epoch': 300, 'trans': False},
            'protocol': {
                'test_candidates': 'complete_target_kg',
                'evaluation_distance': 'faiss_squared_l2',
                'validation_access': 'none',
                'selection_protocol': 'test_best',
                'test_evaluations': 300,
            },
            'test_history': history,
            'best_test': history[0],
            'best_epoch': 0,
            'trained_epochs': 300,
            'stopped_early': False,
        }

    def write(self, payload):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / 'result.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        return temporary, path

    def test_accepts_only_complete_strict_result(self):
        temporary, path = self.write(self.payload())
        try:
            _, failures, evidence = audit(path, 0.884)
            self.assertEqual(
                failures,
                [
                    'final checkpoint lacks a matching independent full-target '
                    'Faiss squared-L2 evaluation'
                ],
            )
            self.assertIsNone(evidence)
        finally:
            temporary.cleanup()

    def test_rejects_csls_or_incomplete_history(self):
        payload = self.payload()
        payload['protocol']['evaluation_distance'] = 'csls'
        payload['test_history'] = payload['test_history'][:-1]
        temporary, path = self.write(payload)
        try:
            _, failures, _ = audit(path, 0.884)
            self.assertIn(
                'final checkpoint lacks a matching independent full-target '
                'Faiss squared-L2 evaluation',
                failures,
            )
            self.assertIn('test history does not contain 300 epochs', failures)
            self.assertIn('test history epoch sequence is not exactly 0 through 299', failures)
        finally:
            temporary.cleanup()

    def test_accepts_matching_independent_l2_checkpoint_report(self):
        payload = self.payload()
        payload['protocol'].pop('evaluation_distance')
        payload['checkpoint'] = '/tmp/strict-pass.pt'
        report = {
            'language': 'zh_en',
            'setting': 'original',
            'candidate_scopes': ['full-target'],
            'evaluation_distance': 'faiss_squared_l2',
            'weight_source': 'online',
            'per_checkpoint': [{
                'checkpoint': '/tmp/strict-pass.pt',
                'selected_epoch': 0,
                'metrics': {'full-target': {
                    'num_queries': 10500,
                    'num_candidates': 19572,
                    'hits1': 0.884,
                    'hits10': 0.97,
                }},
            }],
        }
        temporary, path = self.write(payload)
        try:
            _, failures, evidence = audit(path, 0.884, checkpoint_reports=[report])
            self.assertEqual(failures, [])
            self.assertEqual(evidence['metrics']['num_candidates'], 19572)
        finally:
            temporary.cleanup()

    def test_rejects_wrong_candidate_count_or_checkpoint_epoch(self):
        payload = self.payload()
        payload['checkpoint'] = '/tmp/strict-pass.pt'
        report = {
            'language': 'zh_en',
            'setting': 'original',
            'candidate_scopes': ['full-target'],
            'evaluation_distance': 'faiss_squared_l2',
            'weight_source': 'online',
            'per_checkpoint': [{
                'checkpoint': '/tmp/strict-pass.pt',
                'selected_epoch': 1,
                'metrics': {'full-target': {
                    'num_queries': 10500,
                    'num_candidates': 19573,
                    'hits1': 0.884,
                    'hits10': 0.97,
                }},
            }],
        }
        temporary, path = self.write(payload)
        try:
            _, failures, evidence = audit(path, 0.884, checkpoint_reports=[report])
            self.assertIn(
                'final checkpoint lacks a matching independent full-target '
                'Faiss squared-L2 evaluation',
                failures,
            )
            self.assertIsNone(evidence)
        finally:
            temporary.cleanup()


if __name__ == '__main__':
    unittest.main()
