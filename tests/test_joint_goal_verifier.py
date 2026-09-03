import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_joint_goal import audit


class JointGoalVerifierTest(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        checkpoint = root / 'best_joint_hits10.pt'
        checkpoint.write_bytes(b'checkpoint')
        history = [
            {
                'epoch': epoch,
                'hits1': 0.88,
                'hits10': 0.97,
                'num_queries': 10500,
                'description_scale': 1.0,
                'learning_rate': 1e-6,
                'temperature': 0.08,
                'reverse_icl_weight': 0.5,
            }
            for epoch in range(300)
        ]
        history[250] = {
            'epoch': 250,
            'hits1': 0.889,
            'hits10': 0.972,
            'num_queries': 10500,
            'description_scale': 1.0,
            'learning_rate': 1e-6,
            'temperature': 0.08,
            'reverse_icl_weight': 0.5,
        }
        payload = {
            'status': 'complete',
            'run_name': 'joint-pass',
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
                'lr': 1e-6,
                'lr_schedule': 'constant',
                'lr_min_ratio': 0.1,
                'lr_step_size': 10,
                'lr_decay': 0.5,
                't': 0.08,
                'temperature_schedule': 'constant',
                'temperature_final': 0.08,
                'temperature_step_epoch': 100,
                'reverse_icl_weight': 0.5,
                'reverse_icl_schedule': 'constant',
                'reverse_icl_final_weight': 0.5,
                'reverse_icl_step_epoch': 100,
            },
            'protocol': {
                'pair_threshold': 1.0,
                'pair_mining': 'l2',
                'warmup': 0,
                'rgat_impl': 'paper-exact-rgat',
                'input_slots': 16,
                'maximum_neighbors': 15,
                'gat_layers': 1,
                'negative_set': 'paper-count',
                'description_scale': 1.0,
                'description_scale_schedule': 'constant',
                'description_scale_final': 1.0,
                'description_scale_step_epoch': 200,
                'lr_schedule_name': 'constant',
                'lr_initial': 1e-6,
                'lr_min_ratio': 0.1,
                'lr_step_size': 10,
                'lr_decay': 0.5,
                'temperature_schedule': 'constant',
                'temperature_initial': 0.08,
                'temperature_final': 0.08,
                'temperature_step_epoch': 100,
                'reverse_icl_weight': 0.5,
                'reverse_icl_schedule': 'constant',
                'reverse_icl_final_weight': 0.5,
                'reverse_icl_step_epoch': 100,
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
            'test_history': history,
            'trained_epochs': 300,
            'stopped_early': False,
            'best_joint_test': history[250],
            'best_joint_epoch': 250,
            'joint_checkpoint': str(checkpoint),
        }
        result_path = root / 'joint-pass.json'
        result_path.write_text(json.dumps(payload), encoding='utf-8')
        report = {
            'language': 'zh_en',
            'setting': 'original',
            'num_checkpoints': 1,
            'candidate_scopes': ['full-target'],
            'evaluation_distance': 'faiss_squared_l2',
            'weight_source': 'online',
            'runtime_device': 'cuda',
            'cuda_visible_devices': '0',
            'visible_cuda_device_count': 1,
            'training_description_scale': 1.0,
            'description_scale': 1.0,
            'description_scale_override': False,
            'per_checkpoint': [{
                'checkpoint': str(checkpoint),
                'selected_epoch': 250,
                'weight_source': 'online',
                'training_description_scale': 1.0,
                'description_scale': 1.0,
                'description_scale_override': False,
                'metrics': {'full-target': {
                    'num_queries': 10500,
                    'num_candidates': 19572,
                    'hits1': 0.889,
                    'hits10': 0.972,
                }},
            }],
        }
        return temporary, result_path, payload, report

    def test_accepts_exact_joint_contract(self):
        temporary, path, _, report = self.fixture()
        try:
            _, failures, evidence = audit(path, report)
            self.assertEqual(failures, [])
            self.assertEqual(evidence['selected_epoch'], 250)
        finally:
            temporary.cleanup()

    def test_rejects_embedding_ensemble_or_low_hits10(self):
        temporary, path, payload, report = self.fixture()
        try:
            report['num_checkpoints'] = 2
            payload['best_joint_test']['hits10'] = 0.9719
            path.write_text(json.dumps(payload), encoding='utf-8')
            _, failures, evidence = audit(path, report)
            self.assertIn('joint checkpoint Hits@10 is below target', failures)
            self.assertIn(
                'joint checkpoint lacks one matching independent online full-target Faiss evaluation',
                failures,
            )
            self.assertIsNone(evidence)
        finally:
            temporary.cleanup()

    def test_rejects_non_paper_training_semantics(self):
        temporary, path, payload, report = self.fixture()
        try:
            payload['protocol']['pair_mining'] = 'csls'
            payload['protocol']['input_slots'] = 15
            payload['protocol']['momentum_init'] = 'independent'
            path.write_text(json.dumps(payload), encoding='utf-8')
            _, failures, _ = audit(path, report)
            self.assertIn('pseudo-pair mining is not Faiss L2 Top-1', failures)
            self.assertIn('paper RGAT input slot count is not 16', failures)
            self.assertIn(
                'momentum encoder was not initialized from the online encoder', failures
            )
        finally:
            temporary.cleanup()

    def test_accepts_explicit_top1_without_threshold(self):
        temporary, path, payload, report = self.fixture()
        try:
            payload['profile'] = 'top1-no-threshold'
            payload['arguments']['profile'] = 'top1-no-threshold'
            payload['protocol']['pair_threshold'] = None
            path.write_text(json.dumps(payload), encoding='utf-8')
            _, failures, evidence = audit(path, report)
            self.assertEqual(failures, [])
            self.assertIsNotNone(evidence)
        finally:
            temporary.cleanup()

    def test_rejects_training_or_evaluation_outside_gpu_zero_to_three(self):
        temporary, path, payload, report = self.fixture()
        try:
            payload['protocol']['cuda_visible_devices'] = '4'
            payload['protocol']['physical_gpu'] = 4
            report['cuda_visible_devices'] = '4'
            path.write_text(json.dumps(payload), encoding='utf-8')
            _, failures, evidence = audit(path, report)
            self.assertIn(
                'training was not scoped to exactly one physical GPU 0-3', failures
            )
            self.assertIn(
                'joint checkpoint lacks one matching independent online full-target Faiss evaluation',
                failures,
            )
            self.assertIsNone(evidence)
        finally:
            temporary.cleanup()

    def test_rejects_diagnostic_description_override_as_strict_evidence(self):
        temporary, path, _, report = self.fixture()
        try:
            report['description_scale'] = 1.25
            report['description_scale_override'] = True
            _, failures, evidence = audit(path, report)
            self.assertIn(
                'joint checkpoint lacks one matching independent online full-target Faiss evaluation',
                failures,
            )
            self.assertIsNone(evidence)
        finally:
            temporary.cleanup()

    def test_accepts_declared_late_description_scale_schedule(self):
        temporary, path, payload, report = self.fixture()
        try:
            payload['arguments']['description_scale_schedule'] = 'step-increase'
            payload['arguments']['description_scale_final'] = 2.25
            payload['protocol']['description_scale_schedule'] = 'step-increase'
            payload['protocol']['description_scale_final'] = 2.25
            for item in payload['test_history']:
                item['description_scale'] = 2.25 if item['epoch'] >= 200 else 1.0
            payload['best_joint_test'] = payload['test_history'][250]
            report['training_description_scale'] = 2.25
            report['description_scale'] = 2.25
            report['per_checkpoint'][0]['training_description_scale'] = 2.25
            report['per_checkpoint'][0]['description_scale'] = 2.25
            path.write_text(json.dumps(payload), encoding='utf-8')
            _, failures, evidence = audit(path, report)
            self.assertEqual(failures, [])
            self.assertIsNotNone(evidence)
        finally:
            temporary.cleanup()

    def test_accepts_and_checks_epoch151_nearfreeze_schedule(self):
        temporary, path, payload, report = self.fixture()
        try:
            arguments = payload['arguments']
            protocol = payload['protocol']
            arguments['description_scale_schedule'] = 'step-increase'
            arguments['description_scale_final'] = 2.5
            arguments['description_scale_step_epoch'] = 151
            arguments['lr'] = 2.5e-6
            arguments['lr_schedule'] = 'single-step'
            arguments['lr_step_size'] = 151
            arguments['lr_decay'] = 0.000000001
            protocol['description_scale_schedule'] = 'step-increase'
            protocol['description_scale_final'] = 2.5
            protocol['description_scale_step_epoch'] = 151
            protocol['lr_schedule_name'] = 'single-step'
            protocol['lr_initial'] = 2.5e-6
            protocol['lr_step_size'] = 151
            protocol['lr_decay'] = 0.000000001
            for item in payload['test_history']:
                if item['epoch'] >= 151:
                    item['description_scale'] = 2.5
                    item['learning_rate'] = 2.5e-15
                else:
                    item['description_scale'] = 1.0
                    item['learning_rate'] = 2.5e-6
            payload['best_joint_test'] = payload['test_history'][250]
            report['training_description_scale'] = 2.5
            report['description_scale'] = 2.5
            report['per_checkpoint'][0]['training_description_scale'] = 2.5
            report['per_checkpoint'][0]['description_scale'] = 2.5
            path.write_text(json.dumps(payload), encoding='utf-8')
            _, failures, evidence = audit(path, report)
            self.assertEqual(failures, [])
            self.assertIsNotNone(evidence)

            payload['test_history'][151]['learning_rate'] = 2.5e-6
            path.write_text(json.dumps(payload), encoding='utf-8')
            _, failures, _ = audit(path, report)
            self.assertIn(
                'test history learning rates do not match the declared schedule',
                failures,
            )
        finally:
            temporary.cleanup()


if __name__ == '__main__':
    unittest.main()
