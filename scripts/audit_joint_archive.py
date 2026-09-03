#!/usr/bin/env python3
"""Audit archived ZH_EN results under the strict joint-metric protocol."""

import argparse
import glob
import json
from pathlib import Path


def strict_reason(payload, expected_epochs, allow_legacy_missing_distance=False):
    arguments = payload.get('arguments', {})
    protocol = payload.get('protocol', {})
    checks = (
        (payload.get('status') == 'complete', 'status'),
        (payload.get('language') == 'zh_en', 'language'),
        (payload.get('setting') == 'original' and not arguments.get('trans', False), 'setting'),
        (payload.get('profile') == 'paper', 'profile'),
        (payload.get('rgat_impl') == 'paper-exact-rgat', 'rgat_impl'),
        (int(arguments.get('epoch', -1)) == expected_epochs, 'epochs'),
        (len(payload.get('test_history', ())) == expected_epochs, 'history'),
        (protocol.get('test_candidates') == 'complete_target_kg', 'candidates'),
        (
            protocol.get('evaluation_distance') == 'faiss_squared_l2'
            or (allow_legacy_missing_distance and 'evaluation_distance' not in protocol),
            'distance',
        ),
        (protocol.get('validation_access') == 'none', 'validation'),
    )
    return [name for passed, name in checks if not passed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-glob', default='out/results/zh_en*.json')
    parser.add_argument('--hits1-floor', type=float, default=0.8887619047619048)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--allow-legacy-missing-distance', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    accepted = []
    rejected = []
    for path_string in sorted(glob.glob(args.results_glob)):
        path = Path(path_string)
        payload = json.loads(path.read_text())
        reasons = strict_reason(
            payload,
            args.epochs,
            allow_legacy_missing_distance=args.allow_legacy_missing_distance,
        )
        if reasons:
            rejected.append({'path': str(path), 'reasons': reasons})
            continue
        eligible = [
            row for row in payload['test_history']
            if float(row['hits1']) >= args.hits1_floor
        ]
        joint = max(
            eligible,
            key=lambda row: (float(row['hits10']), float(row['hits1']), -int(row['epoch'])),
        ) if eligible else None
        accepted.append({
            'path': str(path),
            'run_name': payload['run_name'],
            'best_test': payload.get('best_test'),
            'joint': joint,
            'eligible_epochs': len(eligible),
            'checkpoint': payload.get('joint_checkpoint'),
            'arguments': payload.get('arguments', {}),
            'evidence_grade': (
                'legacy-protocol-compatible'
                if 'evaluation_distance' not in payload.get('protocol', {})
                else 'strict-protocol-recorded'
            ),
        })
    accepted.sort(
        key=lambda row: (
            -1.0 if row['joint'] is None else float(row['joint']['hits10']),
            -1.0 if row['joint'] is None else float(row['joint']['hits1']),
        ),
        reverse=True,
    )
    report = {
        'hits1_floor': args.hits1_floor,
        'allow_legacy_missing_distance': args.allow_legacy_missing_distance,
        'accepted_count': len(accepted),
        'rejected_count': len(rejected),
        'accepted': accepted,
        'rejected': rejected,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('accepted={} rejected={}'.format(len(accepted), len(rejected)))
    for row in accepted:
        print(
            '{} joint={} eligible_epochs={} best_test={}'.format(
                row['run_name'], row['joint'], row['eligible_epochs'], row['best_test']
            )
        )


if __name__ == '__main__':
    main()
