#!/usr/bin/env python3
"""Audit full-target ICLEA rank errors under exact Faiss squared L2.

This is a diagnostic only.  It never changes checkpoint metadata and cannot be
used by the strict joint-goal verifier as replacement evaluation evidence.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.ICLEA import MyEmbedder, Trainer, VOCAB_SIZE  # noqa: E402
from scripts.evaluate_checkpoints import trainer_argv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--scales', default='1.0,2.5')
    parser.add_argument('--max-rank', type=int, default=100)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def _quantiles(values):
    if len(values) == 0:
        return {'count': 0}
    return {
        'count': int(len(values)),
        'min': float(np.min(values)),
        'p25': float(np.percentile(values, 25)),
        'median': float(np.median(values)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
        'max': float(np.max(values)),
    }


def exact_rank_profile(queries, candidates, target, max_rank=100):
    """Return an exact top-k rank profile and private arrays for transitions."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    candidates = np.ascontiguousarray(candidates, dtype=np.float32)
    target = np.asarray(target, dtype=np.int64)
    if queries.ndim != 2 or candidates.ndim != 2:
        raise ValueError('queries and candidates must be matrices')
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError('query and candidate dimensions differ')
    if len(queries) != len(target):
        raise ValueError('one target is required per query')
    if len(candidates) == 0:
        raise ValueError('candidate matrix must not be empty')
    if np.any(target < 0) or np.any(target >= len(candidates)):
        raise ValueError('target row is outside the candidate matrix')
    if max_rank < 10:
        raise ValueError('max_rank must be at least 10')

    search_k = min(int(max_rank), len(candidates))
    index = faiss.IndexFlatL2(candidates.shape[1])
    index.add(candidates)
    distances, neighbors = index.search(queries, search_k)
    matches = neighbors == target[:, None]
    found = matches.any(axis=1)
    ranks = np.full(len(target), search_k + 1, dtype=np.int64)
    ranks[found] = np.argmax(matches[found], axis=1) + 1

    target_delta = queries - candidates[target]
    true_distances = np.sum(target_delta * target_delta, axis=1, dtype=np.float32)
    distance_to_top1 = true_distances - distances[:, 0]
    distance_to_top10 = true_distances - distances[:, min(9, search_k - 1)]
    top10_miss = ranks > 10
    top10_counts = np.bincount(
        neighbors[:, :min(10, search_k)].reshape(-1), minlength=len(candidates)
    )
    nonzero_hubness = top10_counts[top10_counts > 0]

    def hits_at(k):
        return float(np.mean(ranks <= min(k, search_k)))

    bins = {
        'rank_1': int(np.sum(ranks == 1)),
        'rank_2_5': int(np.sum((ranks >= 2) & (ranks <= 5))),
        'rank_6_10': int(np.sum((ranks >= 6) & (ranks <= 10))),
        'rank_11_20': int(np.sum((ranks >= 11) & (ranks <= min(20, search_k)))),
        'rank_21_50': int(np.sum((ranks >= 21) & (ranks <= min(50, search_k)))),
        'rank_51_max': int(np.sum((ranks >= 51) & (ranks <= search_k))),
        'beyond_max_rank': int(np.sum(~found)),
    }
    report = {
        'num_queries': int(len(target)),
        'num_candidates': int(len(candidates)),
        'max_rank': int(search_k),
        'hits1': hits_at(1),
        'hits10': hits_at(10),
        'hits50': hits_at(50),
        'hits_at_max_rank': hits_at(search_k),
        'rank_bins': bins,
        'target_minus_top1_squared_l2': _quantiles(distance_to_top1),
        'h10_miss_target_minus_rank10_squared_l2': _quantiles(
            distance_to_top10[top10_miss]
        ),
        'top10_hubness': {
            'unique_candidates': int(np.sum(top10_counts > 0)),
            'max_query_occurrences': int(np.max(top10_counts)),
            'p95_nonzero_query_occurrences': (
                float(np.percentile(nonzero_hubness, 95))
                if len(nonzero_hubness)
                else 0.0
            ),
        },
    }
    private = {
        'neighbors': neighbors,
        'ranks': ranks,
        'hits1': ranks <= 1,
        'hits10': ranks <= 10,
    }
    return report, private


def encode_at_scale(trainer, scale):
    scale = float(scale)
    trainer.args.description_scale = scale
    trainer.model.args.description_scale = scale
    trainer.model.description_scale = scale
    with torch.no_grad():
        trainer.model.eval()
        return trainer._encode_both()


def transition_report(before, after, source_entity_ids, target_entity_ids):
    def transition(before_mask, after_mask):
        rescued = np.flatnonzero(~before_mask & after_mask)
        lost = np.flatnonzero(before_mask & ~after_mask)
        return {
            'rescued_count': int(len(rescued)),
            'lost_count': int(len(lost)),
            'net_count': int(len(rescued) - len(lost)),
            'rescued_source_entity_ids': source_entity_ids[rescued].astype(int).tolist(),
            'rescued_target_entity_ids': target_entity_ids[rescued].astype(int).tolist(),
            'lost_source_entity_ids': source_entity_ids[lost].astype(int).tolist(),
            'lost_target_entity_ids': target_entity_ids[lost].astype(int).tolist(),
        }

    before_rank = before['ranks']
    after_rank = after['ranks']
    return {
        'hits1': transition(before['hits1'], after['hits1']),
        'hits10': transition(before['hits10'], after['hits10']),
        'rank_improved_within_audit_window': int(np.sum(after_rank < before_rank)),
        'rank_worsened_within_audit_window': int(np.sum(after_rank > before_rank)),
        'rank_unchanged_within_audit_window': int(np.sum(after_rank == before_rank)),
    }


def main():
    cli = parse_args()
    if cli.device == 'cpu' and torch.cuda.is_available():
        raise RuntimeError('CPU diagnostics require CUDA_VISIBLE_DEVICES=""')
    scales = [float(value) for value in cli.scales.split(',')]
    if len(scales) < 1 or any(value <= 0.0 for value in scales):
        raise ValueError('all scales must be positive')
    if len(set(scales)) != len(scales):
        raise ValueError('scales must be distinct')

    checkpoint = cli.checkpoint.resolve()
    payload = torch.load(str(checkpoint), map_location='cpu')
    original_argv = sys.argv
    sys.argv = trainer_argv(payload['arguments'])
    try:
        trainer = Trainer(training=False)
    finally:
        sys.argv = original_argv
    trainer.device = torch.device(cli.device)
    trainer.model = MyEmbedder(trainer.args, VOCAB_SIZE).to(trainer.device)
    trainer.model.load_state_dict(payload['online_model'])
    source, target = trainer._link_indices(
        trainer.dataset1, trainer.dataset2, trainer.test_link_tensor
    )
    link_ids = trainer.test_link_tensor.detach().cpu().numpy()

    profiles = {}
    private = {}
    for scale in scales:
        vector_1, vector_2 = encode_at_scale(trainer, scale)
        profile, arrays = exact_rank_profile(
            vector_1[source], vector_2, target, max_rank=cli.max_rank
        )
        profiles[str(scale)] = profile
        private[str(scale)] = arrays

    transitions = {}
    for before_scale, after_scale in zip(scales[:-1], scales[1:]):
        key = '{}->{}'.format(before_scale, after_scale)
        transitions[key] = transition_report(
            private[str(before_scale)],
            private[str(after_scale)],
            link_ids[:, 0],
            link_ids[:, 1],
        )

    report = {
        'diagnostic_only': True,
        'checkpoint': str(checkpoint),
        'selected_epoch': int(payload['epoch']),
        'weight_source': 'online',
        'candidate_scope': 'full-target',
        'evaluation_distance': 'faiss_squared_l2',
        'faiss_index': 'IndexFlatL2',
        'runtime_device': cli.device,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'scales': scales,
        'profiles': profiles,
        'transitions': transitions,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = cli.output.with_suffix(cli.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temporary.replace(cli.output)
    print(json.dumps({
        'output': str(cli.output.resolve()),
        'profiles': {
            key: {'hits1': value['hits1'], 'hits10': value['hits10']}
            for key, value in profiles.items()
        },
        'transitions': transitions,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
