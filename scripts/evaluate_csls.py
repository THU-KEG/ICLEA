#!/usr/bin/env python3
"""Evaluate a saved ICLEA encoder with plain cosine and CSLS reranking."""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.ICLEA import MyEmbedder, Trainer, VOCAB_SIZE  # noqa: E402
from scripts.evaluate_checkpoints import normalize_rows, trainer_argv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--csls-k', type=int, default=10)
    parser.add_argument(
        '--csls-k-values',
        default='',
        help='Optional comma-separated diagnostic sweep; --csls-k remains the primary value.',
    )
    parser.add_argument('--candidate-pool', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.csls_k < 1 or args.candidate_pool < 10:
        parser.error('--csls-k must be positive and --candidate-pool must be at least 10')
    args.csls_k_values = sorted(
        {args.csls_k}
        | {int(value) for value in args.csls_k_values.split(',') if value.strip()}
    )
    if args.csls_k_values[0] < 1:
        parser.error('every --csls-k-values entry must be positive')
    if args.csls_k_values[-1] > args.candidate_pool:
        parser.error('CSLS k cannot exceed --candidate-pool')
    return args


def retrieval_metrics(neighbors, target):
    return {
        'hits1': float((neighbors[:, 0] == target).mean()),
        'hits10': float((neighbors[:, :10] == target[:, None]).any(axis=1).mean()),
        'num_queries': int(len(target)),
    }


def main():
    cli = parse_args()
    checkpoint = cli.checkpoint.resolve()
    payload = torch.load(str(checkpoint), map_location='cpu')
    arguments = payload['arguments']
    original_argv = sys.argv
    sys.argv = trainer_argv(arguments)
    try:
        trainer = Trainer(training=False)
    finally:
        sys.argv = original_argv
    trainer.model = MyEmbedder(trainer.args, VOCAB_SIZE).to(trainer.device)
    trainer.model.load_state_dict(payload['online_model'])
    vector1, vector2 = trainer._encode_both()
    vector1 = normalize_rows(vector1)
    vector2 = normalize_rows(vector2)
    source, target = trainer._link_indices(
        trainer.dataset1, trainer.dataset2, trainer.test_link_tensor
    )
    queries = np.ascontiguousarray(vector1[source].astype(np.float32))
    candidates = np.ascontiguousarray(vector2.astype(np.float32))

    target_index = faiss.IndexFlatIP(candidates.shape[1])
    target_index.add(candidates)
    similarities, pool = target_index.search(queries, cli.candidate_pool)
    source_index = faiss.IndexFlatIP(vector1.shape[1])
    source_index.add(np.ascontiguousarray(vector1.astype(np.float32)))
    maximum_k = max(cli.csls_k_values)
    reverse_similarities, _ = source_index.search(candidates, maximum_k)

    csls_sweep = {}
    primary_neighbors = None
    for k in cli.csls_k_values:
        query_density = similarities[:, :k].mean(axis=1)
        candidate_density = reverse_similarities[:, :k].mean(axis=1)
        csls_scores = 2.0 * similarities - query_density[:, None] - candidate_density[pool]
        order = np.argsort(-csls_scores, axis=1)
        csls_neighbors = np.take_along_axis(pool, order, axis=1)
        csls_sweep[str(k)] = retrieval_metrics(csls_neighbors, target)
        if k == cli.csls_k:
            primary_neighbors = csls_neighbors
    report = {
        'checkpoint': str(checkpoint),
        'selected_epoch': int(payload['epoch']),
        'candidate_scope': 'complete_target_kg',
        'plain_cosine': retrieval_metrics(pool, target),
        'csls': retrieval_metrics(primary_neighbors, target),
        'csls_sweep': csls_sweep,
        'csls_k': cli.csls_k,
        'candidate_pool': cli.candidate_pool,
        'note': 'CSLS is a diagnostic retrieval variant and is not the paper L2 metric.',
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = cli.output.with_suffix(cli.output.suffix + '.tmp')
        temporary.write_text(rendered + '\n', encoding='utf-8')
        temporary.replace(cli.output)
    print(rendered)


if __name__ == '__main__':
    main()
