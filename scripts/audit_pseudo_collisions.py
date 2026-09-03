#!/usr/bin/env python3
"""Measure many-to-one collisions in raw-feature Top-1 pseudo alignments."""

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--translated', action='store_true')
    parser.add_argument('--threshold', type=float, default=1.0)
    return parser.parse_args()


def load_vectors(root, side, translated):
    name_prefix = 'transLaBSE_name_emb_' if translated else 'raw_LaBSE_emb_'
    desc_prefix = 'trans_desc_LaBSE_emb_' if translated else 'desc_LaBSE_emb_'
    parts = []
    for filename in (name_prefix + side + '.pkl', desc_prefix + side + '.pkl'):
        with (root / filename).open('rb') as f:
            values = pickle.load(f)
        rows = np.stack([
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in values.values()
        ])
        rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        parts.append(rows)
    rows = np.concatenate(parts, axis=1)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def direction(query, candidate, threshold):
    index = faiss.IndexFlatL2(candidate.shape[1])
    index.add(np.ascontiguousarray(candidate))
    distances, neighbors = index.search(np.ascontiguousarray(query), 1)
    selected = neighbors[:, 0]
    counts = np.bincount(selected, minlength=len(candidate))
    used = counts[counts > 0]
    return {
        'queries': int(len(query)),
        'unique_targets': int(len(used)),
        'collision_queries': int(np.maximum(used - 1, 0).sum()),
        'targets_with_multiple_queries': int((used > 1).sum()),
        'maximum_multiplicity': int(used.max()),
        'threshold_pass': int((distances[:, 0] < threshold).sum()),
        'multiplicity_percentiles': {
            key: float(value)
            for key, value in zip(
                ('p50', 'p90', 'p95', 'p99'),
                np.percentile(used, (50, 90, 95, 99)),
            )
        },
    }


def main():
    args = parse_args()
    root = args.data_dir.resolve()
    side1 = load_vectors(root, '1', args.translated)
    side2 = load_vectors(root, '2', args.translated)
    report = {
        'setting': 'translated' if args.translated else 'original',
        'threshold': args.threshold,
        '1_to_2': direction(side1, side2, args.threshold),
        '2_to_1': direction(side2, side1, args.threshold),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
