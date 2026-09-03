#!/usr/bin/env python3
"""Align ICLEA loss, pseudo-pair collisions, and retrieval metrics by epoch."""

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path


EPOCH_RE = re.compile(r"Epoch (\d+) learning_rate=([0-9.eE+-]+)")
LOSS_RE = re.compile(
    r"epoch=(\d+) batch=(\d+) step=(\d+) loss=([0-9.eE+-]+)"
)
PSEUDO_RE = re.compile(
    r"Pseudo pairs: (\d+) \+ (\d+) unique_targets=(\d+)\+(\d+) "
    r"collisions=(\d+)\+(\d+)"
)
METRIC_RE = re.compile(
    r"Test: epoch=(\d+).*?Hits@1=([0-9.]+).*?Hits@10=([0-9.]+)"
)


def parse_log(path):
    epochs = {}
    current_epoch = None
    with open(path, 'r', errors='ignore') as handle:
        for line in handle:
            match = EPOCH_RE.search(line)
            if match:
                current_epoch = int(match.group(1))
                row = epochs.setdefault(current_epoch, {'epoch': current_epoch})
                row['learning_rate'] = float(match.group(2))

            match = LOSS_RE.search(line)
            if match:
                epoch, batch, step = (int(value) for value in match.groups()[:3])
                loss = float(match.group(4))
                row = epochs.setdefault(epoch, {'epoch': epoch})
                row.setdefault('_loss_samples', []).append({
                    'batch': batch,
                    'step': step,
                    'loss': loss,
                })

            match = PSEUDO_RE.search(line)
            if match:
                if current_epoch is None:
                    raise ValueError('pseudo-pair line appeared before an epoch header')
                pair_1, pair_2, unique_1, unique_2, collision_1, collision_2 = (
                    int(value) for value in match.groups()
                )
                total_pairs = pair_1 + pair_2
                total_collisions = collision_1 + collision_2
                row = epochs.setdefault(current_epoch, {'epoch': current_epoch})
                row['pseudo_pairs'] = [pair_1, pair_2]
                row['pseudo_unique_targets'] = [unique_1, unique_2]
                row['pseudo_collisions'] = [collision_1, collision_2]
                row['pseudo_collision_rate'] = (
                    float(total_collisions) / total_pairs if total_pairs else 0.0
                )

            match = METRIC_RE.search(line)
            if match:
                epoch = int(match.group(1))
                row = epochs.setdefault(epoch, {'epoch': epoch})
                row['hits1'] = float(match.group(2))
                row['hits10'] = float(match.group(3))

    rows = []
    for epoch in sorted(epochs):
        row = epochs[epoch]
        samples = row.pop('_loss_samples', [])
        if samples:
            losses = [item['loss'] for item in samples]
            row['loss_sample_count'] = len(losses)
            row['loss_sample_mean'] = sum(losses) / len(losses)
            row['loss_sample_min'] = min(losses)
            row['loss_sample_max'] = max(losses)
            row['loss_first_step'] = samples[0]['step']
            row['loss_last_step'] = samples[-1]['step']
        rows.append(row)
    return rows


def pearson(rows, x_key, y_key):
    pairs = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows if x_key in row and y_key in row
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return None if denominator == 0.0 else numerator / denominator


def window_summary(rows, size=25):
    windows = []
    metric_rows = [row for row in rows if 'hits1' in row and 'hits10' in row]
    if not metric_rows:
        return windows
    first = metric_rows[0]['epoch']
    last = metric_rows[-1]['epoch']
    for start in range((first // size) * size, last + 1, size):
        selected = [row for row in metric_rows if start <= row['epoch'] < start + size]
        if not selected:
            continue
        summary = {
            'epoch_start': start,
            'epoch_end': start + size - 1,
            'metric_count': len(selected),
            'mean_hits1': sum(row['hits1'] for row in selected) / len(selected),
            'mean_hits10': sum(row['hits10'] for row in selected) / len(selected),
            'max_hits1': max(row['hits1'] for row in selected),
            'max_hits10': max(row['hits10'] for row in selected),
        }
        collision = [row['pseudo_collision_rate'] for row in selected
                     if 'pseudo_collision_rate' in row]
        loss = [row['loss_sample_mean'] for row in selected
                if 'loss_sample_mean' in row]
        if collision:
            summary['mean_pseudo_collision_rate'] = sum(collision) / len(collision)
        if loss:
            summary['mean_sampled_loss'] = sum(loss) / len(loss)
        windows.append(summary)
    return windows


def summarize_run(path, hits1_floor):
    rows = parse_log(path)
    metric_rows = [row for row in rows if 'hits1' in row and 'hits10' in row]
    if not metric_rows:
        raise ValueError('no test metrics found in {}'.format(path))
    best_h1 = max(metric_rows, key=lambda row: (row['hits1'], row['hits10'], -row['epoch']))
    best_h10 = max(metric_rows, key=lambda row: (row['hits10'], row['hits1'], -row['epoch']))
    eligible = [row for row in metric_rows if row['hits1'] >= hits1_floor]
    best_joint = (
        max(eligible, key=lambda row: (row['hits10'], row['hits1'], -row['epoch']))
        if eligible else None
    )
    return {
        'run_name': Path(path).stem,
        'log': str(Path(path).resolve()),
        'metric_count': len(metric_rows),
        'latest_epoch': metric_rows[-1]['epoch'],
        'best_hits1': best_h1,
        'best_hits10': best_h10,
        'best_joint': best_joint,
        'correlations': {
            'sampled_loss_vs_hits1_pearson': pearson(rows, 'loss_sample_mean', 'hits1'),
            'sampled_loss_vs_hits10_pearson': pearson(rows, 'loss_sample_mean', 'hits10'),
            'pseudo_collision_rate_vs_hits1_pearson': pearson(
                rows, 'pseudo_collision_rate', 'hits1'
            ),
            'pseudo_collision_rate_vs_hits10_pearson': pearson(
                rows, 'pseudo_collision_rate', 'hits10'
            ),
        },
        'windows': window_summary(rows),
        'epochs': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--hits1-floor', type=float, default=0.8887619047619048)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.root, args.tag, 'zh_en*.log')))
    if not paths:
        raise SystemExit('no logs matched tag {}'.format(args.tag))
    report = {
        'tag': args.tag,
        'root': args.root,
        'hits1_floor': args.hits1_floor,
        'correlation_caveat': (
            'Pearson values are descriptive associations across checkpoints, not causal evidence. '
            'Loss is sampled every 200 optimizer steps rather than averaged over all batches.'
        ),
        'runs': [summarize_run(path, args.hits1_floor) for path in paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temporary.replace(args.output)
    for run in report['runs']:
        print('{} latest={} best_h10={} best_joint={} correlations={}'.format(
            run['run_name'], run['latest_epoch'], run['best_hits10'].get('hits10'),
            None if run['best_joint'] is None else run['best_joint'].get('hits10'),
            run['correlations'],
        ))


if __name__ == '__main__':
    main()
