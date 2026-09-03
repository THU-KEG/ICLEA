#!/usr/bin/env python3
"""Evaluate archived name/description embeddings and simple score-level mixtures."""

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
    parser.add_argument('--weights', default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.weights = [float(value) for value in args.weights.split(',')]
    return args


def load_pickle(path):
    with path.open('rb') as f:
        return pickle.load(f)


def vector(value):
    return np.asarray(value, dtype=np.float32).reshape(-1)


def normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def load_side(root, side, translated):
    name_file = (
        'transLaBSE_name_emb_{}.pkl'.format(side)
        if translated else 'raw_LaBSE_emb_{}.pkl'.format(side)
    )
    desc_file = (
        'trans_desc_LaBSE_emb_{}.pkl'.format(side)
        if translated else 'desc_LaBSE_emb_{}.pkl'.format(side)
    )
    names = load_pickle(root / name_file)
    descriptions = load_pickle(root / desc_file)
    ids = [int(value) for value in names.keys()]
    if set(ids) != {int(value) for value in descriptions.keys()}:
        raise ValueError('Name and description entity IDs differ for side {}'.format(side))
    name_rows = normalize([vector(names[value]) for value in names.keys()])
    desc_rows = normalize([vector(descriptions[value]) for value in names.keys()])
    return ids, name_rows, desc_rows


def read_links(path):
    links = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            left, right = line.rstrip('\n').split('\t')
            links.append((int(left), int(right)))
    return links


def metrics(query, candidate, target):
    index = faiss.IndexFlatIP(candidate.shape[1])
    index.add(np.ascontiguousarray(candidate))
    _, neighbors = index.search(np.ascontiguousarray(query), 10)
    return {
        'hits1': float((neighbors[:, 0] == target).mean()),
        'hits10': float((neighbors == target[:, None]).any(axis=1).mean()),
        'num_queries': int(len(target)),
    }


def main():
    args = parse_args()
    root = args.data_dir.resolve()
    ids1, names1, desc1 = load_side(root, '1', args.translated)
    ids2, names2, desc2 = load_side(root, '2', args.translated)
    index1 = {entity_id: row for row, entity_id in enumerate(ids1)}
    index2 = {entity_id: row for row, entity_id in enumerate(ids2)}
    links = read_links(root / 'test.ref')
    source = np.asarray([index1[left] for left, _ in links], dtype=np.int64)
    target = np.asarray([index2[right] for _, right in links], dtype=np.int64)
    subset_target = np.arange(len(links), dtype=np.int64)
    report = {'setting': 'translated' if args.translated else 'original', 'weights': []}

    for name_weight in args.weights:
        if not 0.0 <= name_weight <= 1.0:
            raise ValueError('Weights must be in [0,1]')
        desc_weight = 1.0 - name_weight
        query = np.concatenate((
            np.sqrt(name_weight) * names1[source],
            np.sqrt(desc_weight) * desc1[source],
        ), axis=1)
        full_candidate = np.concatenate((
            np.sqrt(name_weight) * names2,
            np.sqrt(desc_weight) * desc2,
        ), axis=1)
        report['weights'].append({
            'name_weight': name_weight,
            'description_weight': desc_weight,
            'full-target': metrics(query, full_candidate, target),
            'test-subset': metrics(query, full_candidate[target], subset_target),
        })

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(rendered + '\n', encoding='utf-8')
        temporary.replace(args.output)
    print('RAW_FEATURE_EVALUATION_JSON=' + json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
