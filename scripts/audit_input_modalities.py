#!/usr/bin/env python3
"""Measure full-target ZH_EN retrieval supplied by archived input modalities."""

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/DBP15K/zh_en'))
    parser.add_argument(
        '--description-weights',
        default='0,0.25,0.5,0.75,1,1.25,1.5,2',
        help='Comma-separated weights applied to the description half of concat.',
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def load_feature(path):
    with path.open('rb') as handle:
        values = pickle.load(handle)
    ids = np.asarray([int(key) for key in values], dtype=np.int64)
    matrix = np.stack(
        [np.asarray(value, dtype=np.float32).reshape(-1) for value in values.values()]
    )
    return ids, matrix


def normalize(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.ascontiguousarray(matrix / np.maximum(norm, 1e-12), dtype=np.float32)


def evaluate(source_ids, source, target_ids, target, pairs):
    source_rows = {int(entity_id): row for row, entity_id in enumerate(source_ids)}
    target_rows = {int(entity_id): row for row, entity_id in enumerate(target_ids)}
    queries = np.asarray([source_rows[int(left)] for left, _ in pairs], dtype=np.int64)
    answers = np.asarray([target_rows[int(right)] for _, right in pairs], dtype=np.int64)
    index = faiss.IndexFlatL2(target.shape[1])
    index.add(target)
    _, neighbors = index.search(source[queries], 10)
    return {
        'num_queries': int(len(pairs)),
        'num_candidates': int(len(target)),
        'hits1': float(np.mean(neighbors[:, 0] == answers)),
        'hits10': float(np.mean(np.any(neighbors == answers[:, None], axis=1))),
    }


def main():
    args = parse_args()
    name_ids_1, names_1 = load_feature(args.data_dir / 'raw_LaBSE_emb_1.pkl')
    name_ids_2, names_2 = load_feature(args.data_dir / 'raw_LaBSE_emb_2.pkl')
    desc_ids_1, descriptions_1 = load_feature(args.data_dir / 'desc_LaBSE_emb_1.pkl')
    desc_ids_2, descriptions_2 = load_feature(args.data_dir / 'desc_LaBSE_emb_2.pkl')
    if not np.array_equal(name_ids_1, desc_ids_1) or not np.array_equal(name_ids_2, desc_ids_2):
        raise ValueError('name and description dictionaries do not use identical entity order')
    pairs = np.loadtxt(str(args.data_dir / 'test.ref'), dtype=np.int64).reshape(-1, 2)
    names_1, names_2 = normalize(names_1), normalize(names_2)
    descriptions_1, descriptions_2 = normalize(descriptions_1), normalize(descriptions_2)
    results = {
        'data_dir': str(args.data_dir.resolve()),
        'name_only': evaluate(name_ids_1, names_1, name_ids_2, names_2, pairs),
        'description_only': evaluate(
            name_ids_1, descriptions_1, name_ids_2, descriptions_2, pairs
        ),
        'weighted_concat': {},
    }
    weights = [float(value) for value in args.description_weights.split(',')]
    for weight in weights:
        source = normalize(np.concatenate((names_1, weight * descriptions_1), axis=1))
        target = normalize(np.concatenate((names_2, weight * descriptions_2), axis=1))
        results['weighted_concat'][str(weight)] = evaluate(
            name_ids_1, source, name_ids_2, target, pairs
        )
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
