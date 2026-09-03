#!/usr/bin/env python3
"""Fail-closed verifier for the strict ZH_EN 88.4% reproduction target."""

import argparse
import json
from pathlib import Path


TARGET = 0.884
EXPECTED_QUERIES = 10500
EXPECTED_CANDIDATES = 19572
EXPECTED_EPOCHS = 300


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('out/results'))
    parser.add_argument('--run-name', action='append', default=[])
    parser.add_argument('--target', type=float, default=TARGET)
    parser.add_argument(
        '--checkpoint-evaluation',
        action='append',
        type=Path,
        default=[],
        help='Independent JSON report emitted by scripts/evaluate_checkpoints.py.',
    )
    return parser.parse_args()


def independent_checkpoint_evidence(payload, reports):
    checkpoint_value = payload.get('checkpoint')
    if not checkpoint_value:
        return None
    checkpoint = str(Path(checkpoint_value).resolve())
    best = payload.get('best_test', {})
    for report in reports:
        if (
            report.get('language') != 'zh_en'
            or report.get('setting') != 'original'
            or report.get('evaluation_distance') != 'faiss_squared_l2'
            or report.get('weight_source') != 'online'
            or 'full-target' not in report.get('candidate_scopes', ())
        ):
            continue
        for row in report.get('per_checkpoint', ()):
            if str(Path(row.get('checkpoint', '')).resolve()) != checkpoint:
                continue
            metrics = row.get('metrics', {}).get('full-target', {})
            valid = (
                int(row.get('selected_epoch', -1)) == int(best.get('epoch', -2))
                and int(metrics.get('num_queries', -1)) == EXPECTED_QUERIES
                and int(metrics.get('num_candidates', -1)) == EXPECTED_CANDIDATES
                and abs(float(metrics.get('hits1', -1.0)) - float(best.get('hits1', -2.0)))
                <= 1e-12
                and abs(float(metrics.get('hits10', -1.0)) - float(best.get('hits10', -2.0)))
                <= 1e-12
            )
            if valid:
                return {
                    'checkpoint': checkpoint,
                    'metrics': metrics,
                    'evaluation_distance': report['evaluation_distance'],
                }
    return None


def audit(path, target, checkpoint_reports=()):
    payload = json.loads(path.read_text(encoding='utf-8'))
    protocol = payload.get('protocol', {})
    arguments = payload.get('arguments', {})
    history = payload.get('test_history', [])
    best = payload.get('best_test', {})
    failures = []
    expected_epoch_sequence = list(range(EXPECTED_EPOCHS))
    observed_epoch_sequence = [int(item.get('epoch', -1)) for item in history]
    calculated_best = (
        max(history, key=lambda item: (float(item['hits1']), float(item['hits10'])))
        if history
        else {}
    )
    best_matches_history = bool(calculated_best) and (
        int(best.get('epoch', -1)) == int(calculated_best.get('epoch', -2))
        and float(best.get('hits1', -1.0)) == float(calculated_best.get('hits1', -2.0))
        and float(best.get('hits10', -1.0)) == float(calculated_best.get('hits10', -2.0))
    )
    external_evidence = independent_checkpoint_evidence(payload, checkpoint_reports)

    expected = (
        (payload.get('status') == 'complete', 'status is not complete'),
        (payload.get('language') == 'zh_en', 'language is not zh_en'),
        (payload.get('setting') == 'original', 'setting is not original multilingual'),
        (not arguments.get('trans', False), '--trans is enabled'),
        (int(arguments.get('epoch', -1)) == EXPECTED_EPOCHS, 'configured epoch is not 300'),
        (len(history) == EXPECTED_EPOCHS, 'test history does not contain 300 epochs'),
        (
            observed_epoch_sequence == expected_epoch_sequence,
            'test history epoch sequence is not exactly 0 through 299',
        ),
        (int(payload.get('trained_epochs', -1)) == EXPECTED_EPOCHS, 'trained_epochs is not 300'),
        (payload.get('stopped_early') is False, 'run stopped early or omits stopped_early=false'),
        (
            protocol.get('test_candidates') == 'complete_target_kg',
            'candidate scope is not the complete target KG',
        ),
        (
            external_evidence is not None,
            'final checkpoint lacks a matching independent full-target Faiss squared-L2 evaluation',
        ),
        (protocol.get('validation_access') == 'none', 'validation data was accessed'),
        (protocol.get('selection_protocol') == 'test_best', 'selection is not test-best'),
        (
            int(protocol.get('test_evaluations', -1)) == EXPECTED_EPOCHS,
            'test evaluation count is not 300',
        ),
        (
            all(int(item.get('num_queries', -1)) == EXPECTED_QUERIES for item in history),
            'one or more evaluations do not contain 10500 test queries',
        ),
        (best_matches_history, 'best_test does not match the maximum test-history item'),
        (
            int(payload.get('best_epoch', -1)) == int(best.get('epoch', -2)),
            'best_epoch does not match best_test.epoch',
        ),
        (float(best.get('hits1', -1.0)) >= target, 'best test Hits@1 is below target'),
    )
    failures.extend(message for passed, message in expected if not passed)
    return payload, failures, external_evidence


def main():
    args = parse_args()
    if args.run_name:
        paths = [args.results_dir / (name + '.json') for name in args.run_name]
    else:
        paths = sorted(args.results_dir.glob('*.json'))
    if not paths:
        raise SystemExit('no result JSON files found')
    checkpoint_reports = [
        json.loads(path.read_text(encoding='utf-8'))
        for path in args.checkpoint_evaluation
    ]

    passing = []
    audited = []
    for path in paths:
        if not path.is_file():
            audited.append({'path': str(path), 'failures': ['result JSON is missing']})
            continue
        payload, failures, external_evidence = audit(
            path, args.target, checkpoint_reports=checkpoint_reports
        )
        row = {
            'path': str(path.resolve()),
            'run_name': payload.get('run_name'),
            'hits1': payload.get('best_test', {}).get('hits1'),
            'failures': failures,
            'independent_checkpoint_evidence': external_evidence,
        }
        audited.append(row)
        if not failures:
            passing.append(row)

    report = {
        'target': args.target,
        'strict_contract': {
            'language': 'zh_en',
            'setting': 'original multilingual',
            'epochs': EXPECTED_EPOCHS,
            'selection': 'test-best without validation access',
            'candidate_scope': 'complete target KG',
            'target_candidates': EXPECTED_CANDIDATES,
            'distance': 'Faiss squared L2',
            'test_queries': EXPECTED_QUERIES,
        },
        'passing': passing,
        'audited': audited,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
