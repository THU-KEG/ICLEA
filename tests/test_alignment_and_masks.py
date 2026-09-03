import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
try:
    import faiss  # noqa: F401
except ImportError:
    sys.modules['faiss'] = types.ModuleType('faiss')

import loader.DBP15KRawDataset_ICLEA as loader_module
from model.ICLEA import (
    AUTHOR_RGAT,
    PAPER_EXACT_RGAT,
    BatchMultiHeadGraphAttention,
    RelationAttention,
    MyEmbedder,
    description_scale_for_epoch,
    learning_rate_for_epoch,
    mine_csls_bidirectional,
    mine_top1,
    metrics_improved,
    validate_rgat_configuration,
    validate_training_gpu_scope,
)
from scripts.evaluate_checkpoints import trainer_argv
from scripts.audit_checkpoint_asymmetric_description_scale import encode_side


def vector(value):
    return [[float(value)] * 768]


class AlignmentAndMaskTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name) / 'DBP15K' / 'zh_en'
        root.mkdir(parents=True)
        self.root = root
        entities = {10: vector(10), 11: vector(11), 12: vector(12)}
        descriptions = {10: vector(20), 11: vector(21), 12: vector(22)}
        relations1 = {0: vector(2), 1: vector(4)}
        relations2 = {2: vector(6)}
        for side in ('1', '2'):
            with (root / ('raw_LaBSE_emb_' + side + '.pkl')).open('wb') as f:
                pickle.dump(entities, f)
            with (root / ('desc_LaBSE_emb_' + side + '.pkl')).open('wb') as f:
                pickle.dump(descriptions, f)
        with (root / 'jape_relation_emb_1.pkl').open('wb') as f:
            pickle.dump(relations1, f)
        with (root / 'jape_relation_emb_2.pkl').open('wb') as f:
            pickle.dump(relations2, f)
        (root / 'jape_hy_1').write_text('0\t10\n1\t11\n2\t12\n', encoding='utf-8')
        (root / 'jape_hy_2').write_text('0\t10\n1\t11\n2\t12\n', encoding='utf-8')
        (root / 'jape_triples_1').write_text(
            '0\t0\t1\n0\t1\t1\n0\t1\t2\n', encoding='utf-8'
        )
        (root / 'jape_triples_2').write_text('0\t2\t1\n', encoding='utf-8')
        self.previous_data_dir = loader_module.DATA_DIR
        loader_module.DATA_DIR = str(Path(self.temp.name))

    def tearDown(self):
        loader_module.DATA_DIR = self.previous_data_dir
        self.temp.cleanup()

    def test_training_gpu_scope_accepts_only_one_allowed_physical_gpu(self):
        self.assertEqual(validate_training_gpu_scope('0', 1), 0)
        self.assertEqual(validate_training_gpu_scope('3', 1), 3)
        for value, count in (
            (None, 1),
            ('', 1),
            ('4', 1),
            ('0,1', 2),
            ('0', 2),
        ):
            with self.subTest(value=value, count=count):
                with self.assertRaises(ValueError):
                    validate_training_gpu_scope(value, count)

    def test_neighbor_relation_slots_and_parallel_relation_mean(self):
        dataset = loader_module.DBP15KRawDataset('zh_en', '1', False)
        self.assertEqual(dataset.relation_padding_idx, 3)
        self.assertEqual(dataset.max_relations_per_neighbor, 2)
        self.assertTrue(torch.all(dataset.id_neighbors_dict[10][1] == 11))
        self.assertTrue(torch.all(dataset.id_neighbors_dict[10][2] == 12))
        self.assertTrue(torch.all(dataset.id_neighbors_relation_dict[10][1] == 3))
        self.assertTrue(torch.all(dataset.id_neighbors_relation_dict[10][2] == 4))
        self.assertEqual(dataset.id_neighbors_relation_ids[10][1].tolist(), [0, 1])
        self.assertEqual(dataset.id_neighbors_relation_mask[10][1].tolist(), [True, True])
        self.assertEqual(dataset.id_neighbors_relation_mask[10][2].tolist(), [True, False])
        self.assertEqual(dataset.id_neighbor_mask[10][:4].tolist(), [False, True, True, False])

    def test_paper_exact_loader_has_center_plus_fifteen_neighbor_slots(self):
        dataset = loader_module.DBP15KRawDataset('zh_en', '1', False, neighbor_size=16)
        self.assertEqual(dataset.id_neighbors_dict[10].shape, (16, 768))
        self.assertEqual(dataset.id_adj_tensor_dict[10].shape, (16, 16))
        self.assertEqual(dataset.id_neighbor_mask[10].shape, (16,))

    def test_relation_attention_excludes_center_and_padding(self):
        attention = RelationAttention(SimpleNamespace())
        with torch.no_grad():
            attention.fc1.weight.zero_()
            attention.fc1.bias.zero_()
            attention.fc2.weight.zero_()
            attention.fc2.bias.zero_()
            attention.w.copy_(torch.eye(1536))
        features = torch.zeros(1, 15, 1536)
        features[:, 0] = 50
        features[:, 1] = 1
        features[:, 2] = 3
        features[:, 3:] = 100
        relations = torch.zeros(1, 15, 768)
        mask = torch.zeros(1, 15, dtype=torch.bool)
        mask[:, 1:3] = True
        output = attention(features, relations, mask)
        self.assertTrue(torch.allclose(output, torch.full_like(output, 2.0)))

    def test_paper_exact_gat_matches_equation_without_inner_tanh(self):
        args = SimpleNamespace(dropout=0.0, rgat_impl=PAPER_EXACT_RGAT)
        attention = BatchMultiHeadGraphAttention(
            args, n_head=1, f_in=1, f_out=1, bias=False
        )
        with torch.no_grad():
            attention.w.fill_(1.0)
            attention.a_src.fill_(1.0)
            attention.a_dst.fill_(1.0)
        features = torch.tensor([[[2.0], [-1.0]]])
        adjacency = torch.ones(1, 2, 2, dtype=torch.bool)
        output = attention(features, adjacency)

        transformed = features.unsqueeze(1)
        scores = attention.leaky_relu(
            transformed.expand(-1, -1, -1, 2)
            + transformed.expand(-1, -1, -1, 2).transpose(2, 3)
        )
        expected = attention.leaky_relu(torch.matmul(torch.softmax(scores, dim=-1), transformed))
        self.assertTrue(torch.allclose(output, expected))

        author = BatchMultiHeadGraphAttention(
            SimpleNamespace(dropout=0.0, rgat_impl=AUTHOR_RGAT),
            n_head=1,
            f_in=1,
            f_out=1,
            bias=False,
        )
        author.load_state_dict(attention.state_dict())
        self.assertFalse(torch.allclose(output, author(features, adjacency)))

    def test_paper_exact_relation_gate_applies_outer_activation(self):
        exact = RelationAttention(SimpleNamespace(rgat_impl=PAPER_EXACT_RGAT))
        author = RelationAttention(SimpleNamespace(rgat_impl=AUTHOR_RGAT))
        with torch.no_grad():
            for attention in (exact, author):
                attention.fc1.weight.zero_()
                attention.fc1.bias.zero_()
                attention.fc2.weight.zero_()
                attention.fc2.bias.zero_()
                attention.w.copy_(-torch.eye(1536))
        features = torch.ones(1, 2, 1536)
        relations = torch.zeros(1, 2, 768)
        mask = torch.ones(1, 2, dtype=torch.bool)
        self.assertTrue(torch.equal(exact(features, relations, mask), torch.zeros(1, 1536)))
        self.assertTrue(torch.equal(author(features, relations, mask), -torch.ones(1, 1536)))

    def test_paper_exact_rejects_non_paper_graph_depth(self):
        with self.assertRaisesRegex(ValueError, 'one-hop GAT'):
            validate_rgat_configuration(
                SimpleNamespace(gat_num=2, rgat_impl=PAPER_EXACT_RGAT)
            )

    def test_checkpoint_evaluator_reconstructs_rgat_variant(self):
        arguments = {
            'language': 'zh_en',
            'model_language': 'zh_en',
            'profile': 'paper',
            'rgat_impl': PAPER_EXACT_RGAT,
            'momentum_init': 'independent',
            'pair_mining': 'csls',
            'icl_beta': 0.85,
            'icl_source_inbatch': 'source',
            'description_scale': 1.25,
            'description_scale_schedule': 'step-increase',
            'description_scale_final': 2.5,
            'description_scale_step_epoch': 150,
        }
        values = trainer_argv(arguments)
        position = values.index('--rgat_impl')
        self.assertEqual(values[position + 1], PAPER_EXACT_RGAT)
        position = values.index('--momentum_init')
        self.assertEqual(values[position + 1], 'independent')
        position = values.index('--pair_mining')
        self.assertEqual(values[position + 1], 'csls')
        position = values.index('--icl_beta')
        self.assertEqual(values[position + 1], '0.85')
        position = values.index('--icl_source_inbatch')
        self.assertEqual(values[position + 1], 'source')
        position = values.index('--description_scale')
        self.assertEqual(values[position + 1], '1.25')
        position = values.index('--description_scale_schedule')
        self.assertEqual(values[position + 1], 'step-increase')
        position = values.index('--description_scale_final')
        self.assertEqual(values[position + 1], '2.5')
        position = values.index('--description_scale_step_epoch')
        self.assertEqual(values[position + 1], '150')

    @unittest.skipUnless(hasattr(sys.modules['faiss'], 'IndexFlatIP'), 'faiss is unavailable')
    def test_csls_mining_returns_squared_l2_for_selected_neighbor(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).numpy()
        candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]).numpy()
        distances, neighbors = mine_top1(
            query, candidates, method='csls', csls_k=1, candidate_pool=3
        )
        self.assertEqual(neighbors.tolist(), [0, 1])
        self.assertTrue((distances == 0.0).all())

        forward, reverse = mine_csls_bidirectional(
            query, candidates, csls_k=1, candidate_pool=3
        )
        self.assertEqual(forward[1].tolist(), neighbors.tolist())
        self.assertTrue((forward[0] == distances).all())
        expected_reverse = mine_top1(
            candidates, query, method='csls', csls_k=1, candidate_pool=2
        )
        self.assertEqual(reverse[1].tolist(), expected_reverse[1].tolist())
        self.assertTrue((reverse[0] == expected_reverse[0]).all())

    def test_learning_rate_is_monotone_and_does_not_reset(self):
        args = SimpleNamespace(
            lr=1e-6, lr_schedule='step', lr_step_size=10, lr_decay=0.5
        )
        self.assertEqual(learning_rate_for_epoch(args, 0), 1e-6)
        self.assertEqual(learning_rate_for_epoch(args, 9), 1e-6)
        self.assertEqual(learning_rate_for_epoch(args, 10), 5e-7)
        self.assertEqual(learning_rate_for_epoch(args, 11), 5e-7)
        self.assertEqual(learning_rate_for_epoch(args, 20), 2.5e-7)

    def test_released_learning_rate_schedule_only_halves_periodic_epochs(self):
        args = SimpleNamespace(
            lr=1e-6, lr_schedule='author-periodic', lr_step_size=10, lr_decay=0.5
        )
        self.assertEqual(learning_rate_for_epoch(args, 8), 1e-6)
        self.assertEqual(learning_rate_for_epoch(args, 9), 5e-7)
        self.assertEqual(learning_rate_for_epoch(args, 10), 1e-6)
        self.assertEqual(learning_rate_for_epoch(args, 19), 5e-7)

    def test_single_step_learning_rate_does_not_repeat(self):
        args = SimpleNamespace(
            lr=3e-6,
            lr_schedule='single-step',
            lr_step_size=100,
            lr_decay=0.5,
        )
        self.assertEqual(learning_rate_for_epoch(args, 99), 3e-6)
        self.assertEqual(learning_rate_for_epoch(args, 100), 1.5e-6)
        self.assertEqual(learning_rate_for_epoch(args, 299), 1.5e-6)

    def test_cosine_learning_rate_reaches_explicit_floor_without_reset(self):
        args = SimpleNamespace(
            lr=2e-6,
            lr_schedule='cosine',
            lr_step_size=10,
            lr_decay=0.5,
            lr_min_ratio=0.1,
            epoch=300,
        )
        values = [learning_rate_for_epoch(args, epoch) for epoch in range(300)]
        self.assertEqual(values[0], 2e-6)
        self.assertAlmostEqual(values[-1], 2e-7)
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))

    def test_temperature_schedules_are_explicit_and_monotone(self):
        from model.ICLEA import temperature_for_epoch

        linear = SimpleNamespace(
            t=0.08,
            temperature_final=0.12,
            temperature_schedule='linear-increase',
            temperature_step_epoch=100,
            epoch=300,
        )
        values = [temperature_for_epoch(linear, epoch) for epoch in range(300)]
        self.assertEqual(values[0], 0.08)
        self.assertAlmostEqual(values[-1], 0.12)
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))

        step = SimpleNamespace(
            t=0.08,
            temperature_final=0.10,
            temperature_schedule='step-increase',
            temperature_step_epoch=100,
            epoch=300,
        )
        self.assertEqual(temperature_for_epoch(step, 99), 0.08)
        self.assertEqual(temperature_for_epoch(step, 100), 0.10)

    def test_reverse_icl_weight_schedules_are_explicit_and_monotone(self):
        from model.ICLEA import reverse_icl_weight_for_epoch

        linear = SimpleNamespace(
            reverse_icl_weight=0.5,
            reverse_icl_final_weight=0.0,
            reverse_icl_schedule='linear-decay',
            reverse_icl_step_epoch=100,
            epoch=300,
        )
        values = [reverse_icl_weight_for_epoch(linear, epoch) for epoch in range(300)]
        self.assertEqual(values[0], 0.5)
        self.assertEqual(values[99], 0.5)
        self.assertEqual(values[100], 0.5)
        self.assertAlmostEqual(values[-1], 0.0)
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))

        step = SimpleNamespace(
            reverse_icl_weight=0.5,
            reverse_icl_final_weight=0.25,
            reverse_icl_schedule='step-decay',
            reverse_icl_step_epoch=100,
            epoch=300,
        )
        self.assertEqual(reverse_icl_weight_for_epoch(step, 99), 0.5)
        self.assertEqual(reverse_icl_weight_for_epoch(step, 100), 0.25)

    def test_description_scale_schedule_can_make_one_late_input_only_step(self):
        args = SimpleNamespace(
            description_scale=1.0,
            description_scale_schedule='step-increase',
            description_scale_final=2.25,
            description_scale_step_epoch=200,
            epoch=300,
        )
        self.assertEqual(description_scale_for_epoch(args, 199), 1.0)
        self.assertEqual(description_scale_for_epoch(args, 200), 2.25)
        args.description_scale_schedule = 'linear-increase'
        self.assertEqual(description_scale_for_epoch(args, 199), 1.0)
        self.assertEqual(description_scale_for_epoch(args, 299), 2.25)

    def test_asymmetric_diagnostic_sets_runtime_model_scale(self):
        model = SimpleNamespace(
            args=SimpleNamespace(description_scale=1.0),
            description_scale=1.0,
            eval=lambda: None,
        )
        trainer = SimpleNamespace(
            args=SimpleNamespace(description_scale=1.0),
            model=model,
        )
        trainer._encode_dataset = lambda dataset: (
            trainer.model.description_scale,
            dataset,
        )
        self.assertEqual(encode_side(trainer, 'kg1', 2.5), (2.5, 'kg1'))
        self.assertEqual(trainer.args.description_scale, 2.5)
        self.assertEqual(trainer.model.args.description_scale, 2.5)

    def test_paper_count_adds_other_positive_batch_rows_as_negatives(self):
        holder = SimpleNamespace(
            args=SimpleNamespace(t=1.0, negative_set='paper-count'),
            criterion=lambda logits: logits,
        )
        pos = torch.eye(3)
        queued = torch.zeros(2, 3)
        logits = MyEmbedder.contrastive_loss(holder, pos, pos, queued)
        self.assertEqual(tuple(logits.shape), (3, 6))
        self.assertTrue(torch.equal(logits[:, 0], torch.ones(3)))
        self.assertTrue(torch.all(logits[torch.arange(3), torch.arange(3) + 1] < -1e20))

    def test_false_negative_mask_excludes_queue_and_duplicate_positive_ids(self):
        holder = SimpleNamespace(
            args=SimpleNamespace(t=1.0, negative_set='paper-count'),
            criterion=lambda logits: logits,
        )
        pos_1 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        pos_2 = pos_1.clone()
        queued = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        logits = MyEmbedder.contrastive_loss(
            holder,
            pos_1,
            pos_2,
            queued,
            positive_key_ids=torch.tensor([7, 7, 9]),
            negative_key_ids=torch.tensor([7, 8]),
        )
        self.assertTrue(torch.all(logits[:2, 1:3] < -1e20))
        self.assertTrue(torch.all(logits[:2, -2] < -1e20))
        self.assertGreater(logits[2, -2].item(), -1e20)

    def test_paper_source_icl_uses_source_inbatch_and_source_queue_ids(self):
        holder = SimpleNamespace(
            args=SimpleNamespace(t=1.0, negative_set='paper-count'),
            criterion=lambda logits: logits,
        )
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        pseudo_target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        source_momentum = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        source_queue = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        logits = MyEmbedder.contrastive_loss(
            holder,
            query,
            pseudo_target,
            source_queue,
            positive_key_ids=torch.tensor([101, 102]),
            negative_key_ids=torch.tensor([11, 99]),
            queue_exclude_ids=torch.tensor([11, 12]),
            in_batch_value=source_momentum,
            in_batch_key_ids=torch.tensor([11, 12, 13]),
            in_batch_exclude_ids=torch.tensor([11, 12]),
        )
        self.assertTrue(torch.all(logits[torch.arange(2), torch.arange(2) + 1] < -1e20))
        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertLess(logits[0, -2].item(), -1e20)
        self.assertGreater(logits[1, -2].item(), -1e20)

    def test_test_best_selection_prefers_hits1_then_hits10(self):
        incumbent = {'hits1': 0.80, 'hits10': 0.90}
        self.assertTrue(metrics_improved({'hits1': 0.81, 'hits10': 0.85}, incumbent))
        self.assertTrue(metrics_improved({'hits1': 0.80, 'hits10': 0.91}, incumbent))
        self.assertFalse(metrics_improved({'hits1': 0.80, 'hits10': 0.89}, incumbent))
        self.assertFalse(metrics_improved({'hits1': 0.79, 'hits10': 0.99}, incumbent))

    def test_joint_selection_prefers_hits10_then_hits1(self):
        from model.ICLEA import joint_metrics_improved

        incumbent = {'hits1': 0.89, 'hits10': 0.96}
        self.assertTrue(joint_metrics_improved({'hits1': 0.89, 'hits10': 0.97}, incumbent))
        self.assertTrue(joint_metrics_improved({'hits1': 0.90, 'hits10': 0.96}, incumbent))
        self.assertFalse(joint_metrics_improved({'hits1': 0.91, 'hits10': 0.95}, incumbent))


if __name__ == '__main__':
    unittest.main()
