#!/usr/bin/env python3
"""Select highest Hits@10 subject to a strict Hits@1 floor from completed runs."""

import argparse
import glob
import json
from pathlib import Path


EXPECTED_QUERIES = 10500
EXPECTED_EPOCHS = 300


def required_run_names(manifest_path):
    lines = Path(manifest_path).read_text().splitlines()
    if not lines or lines[0].split('\t', 1)[0].strip() != 'run_name':
        raise ValueError('invalid run manifest header: {}'.format(manifest_path))
    names = [
        line.split('\t', 1)[0].strip()
        for line in lines[1:]
        if line.strip()
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError('run manifest must contain unique run names: {}'.format(manifest_path))
    return names


def expected_description_scale(arguments, epoch):
    initial = float(arguments.get('description_scale', -1.0))
    final = float(arguments.get('description_scale_final', initial))
    schedule = arguments.get('description_scale_schedule', 'constant')
    step = int(arguments.get('description_scale_step_epoch', 200))
    if schedule == 'constant':
        return initial
    if schedule == 'step-increase':
        return final if epoch >= step else initial
    if schedule == 'linear-increase':
        progress = max(0.0, float(epoch - step) / max(EXPECTED_EPOCHS - 1 - step, 1))
        return initial + (final - initial) * progress
    return -1.0


def training_contract_reasons(payload, hits1_floor):
    arguments = payload.get('arguments', {})
    protocol = payload.get('protocol', {})
    history = payload.get('test_history', ())
    profile = payload.get('profile')
    epochs = int(arguments.get('epoch', -1))
    observed_epochs = [int(row.get('epoch', -1)) for row in history]
    resource_recorded = any(
        key in protocol
        for key in ('cuda_visible_devices', 'visible_cuda_device_count', 'physical_gpu')
    )
    description_scale_recorded = (
        'description_scale' in protocol or 'description_scale' in arguments
    )
    checks = (
        (payload.get('status') == 'complete', 'status'),
        (payload.get('language') == 'zh_en', 'language'),
        (payload.get('setting') == 'original' and not arguments.get('trans', False), 'setting'),
        (
            profile in ('paper', 'top1-no-threshold')
            and arguments.get('profile') == profile,
            'profile',
        ),
        (
            payload.get('rgat_impl') == 'paper-exact-rgat'
            and arguments.get('rgat_impl') == 'paper-exact-rgat'
            and protocol.get('rgat_impl') == 'paper-exact-rgat',
            'rgat_impl',
        ),
        (int(protocol.get('input_slots', -1)) == 16, 'input_slots'),
        (int(protocol.get('maximum_neighbors', -1)) == 15, 'maximum_neighbors'),
        (int(protocol.get('gat_layers', -1)) == 1, 'gat_layers'),
        (protocol.get('pair_mining') == 'l2', 'pair_mining'),
        (
            (
                profile == 'paper'
                and protocol.get('pair_threshold') is not None
                and abs(float(protocol['pair_threshold']) - 1.0) <= 1e-12
            )
            or (profile == 'top1-no-threshold' and protocol.get('pair_threshold') is None),
            'pair_threshold',
        ),
        (int(protocol.get('warmup', -1)) == 0, 'warmup'),
        (protocol.get('negative_set') == 'paper-count', 'negative_set'),
        (
            not description_scale_recorded
            or (
                float(protocol.get('description_scale', -1.0)) > 0.0
                and abs(
                    float(protocol.get('description_scale'))
                    - float(arguments.get('description_scale', -2.0))
                ) <= 1e-12
                and protocol.get('description_scale_schedule', 'constant')
                == arguments.get('description_scale_schedule', 'constant')
                and abs(
                    float(protocol.get('description_scale_final', -1.0))
                    - float(
                        arguments.get(
                            'description_scale_final',
                            arguments.get('description_scale', -2.0),
                        )
                    )
                ) <= 1e-12
                and int(protocol.get('description_scale_step_epoch', -1))
                == int(arguments.get('description_scale_step_epoch', 200))
                and all(
                    abs(
                        float(row.get('description_scale', -1.0))
                        - expected_description_scale(arguments, int(row.get('epoch', -1)))
                    ) <= 1e-12
                    for row in history
                )
            ),
            'description_scale',
        ),
        (protocol.get('momentum_init') == 'copy-online', 'momentum_init'),
        (epochs == EXPECTED_EPOCHS, 'epochs'),
        (len(history) == EXPECTED_EPOCHS, 'history_length'),
        (observed_epochs == list(range(EXPECTED_EPOCHS)), 'history_epochs'),
        (
            all(int(row.get('num_queries', -1)) == EXPECTED_QUERIES for row in history),
            'num_queries',
        ),
        (int(payload.get('trained_epochs', -1)) == EXPECTED_EPOCHS, 'trained_epochs'),
        (payload.get('stopped_early') is False, 'stopped_early'),
        (protocol.get('test_candidates') == 'complete_target_kg', 'test_candidates'),
        (protocol.get('evaluation_distance') == 'faiss_squared_l2', 'distance'),
        (protocol.get('validation_access') == 'none', 'validation'),
        (
            not resource_recorded
            or (
                str(protocol.get('cuda_visible_devices')) in ('0', '1', '2', '3')
                and int(protocol.get('visible_cuda_device_count', -1)) == 1
                and int(protocol.get('physical_gpu', -1))
                == int(protocol.get('cuda_visible_devices', -2))
            ),
            'gpu_scope',
        ),
        (protocol.get('selection_protocol') == 'test_best', 'selection'),
        (int(protocol.get('test_evaluations', -1)) == EXPECTED_EPOCHS, 'evaluations'),
        (
            arguments.get('joint_hits1_floor') is not None
            and abs(float(arguments['joint_hits1_floor']) - hits1_floor) <= 1e-12,
            'hits1_floor',
        ),
    )
    reasons = [name for passed, name in checks if not passed]
    eligible_history = [row for row in history if float(row.get('hits1', -1.0)) >= hits1_floor]
    calculated = max(
        eligible_history,
        key=lambda row: (float(row['hits10']), float(row['hits1']), -int(row['epoch'])),
    ) if eligible_history else None
    selected = payload.get('best_joint_test')
    if calculated is None:
        if selected is not None or payload.get('best_joint_epoch') is not None:
            reasons.append('joint_selection_without_eligible_epoch')
    elif (
        selected is None
        or int(selected.get('epoch', -1)) != int(calculated['epoch'])
        or float(selected.get('hits1', -1.0)) != float(calculated['hits1'])
        or float(selected.get('hits10', -1.0)) != float(calculated['hits10'])
        or int(payload.get('best_joint_epoch', -1)) != int(calculated['epoch'])
    ):
        reasons.append('joint_selection_mismatch')
    return reasons


def select_joint_payload(payloads, hits1_floor):
    complete = [
        payload
        for payload in payloads
        if not training_contract_reasons(payload, hits1_floor)
    ]
    eligible = [
        payload
        for payload in complete
        if payload.get('best_joint_test')
        and payload['best_joint_test']['hits1'] >= hits1_floor
    ]
    if not eligible:
        raise ValueError('no completed checkpoint meets the Hits@1 floor')
    return max(
        eligible,
        key=lambda payload: (
            payload['best_joint_test']['hits10'],
            payload['best_joint_test']['hits1'],
            -payload['best_joint_epoch'],
        ),
    ), complete, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', action='append', required=True)
    parser.add_argument('--hits1-floor', type=float, required=True)
    parser.add_argument('--hits10-target', type=float, required=True)
    parser.add_argument('--minimum-complete', type=int, default=1)
    parser.add_argument(
        '--required-run-manifest',
        action='append',
        default=[],
        type=Path,
        help='Require every run named by this TSV manifest to pass the training contract.',
    )
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    paths = []
    for tag in args.tag:
        paths.extend(glob.glob('out/results/*_{}.json'.format(tag)))
    paths = sorted(set(paths))
    payloads = [json.loads(Path(path).read_text()) for path in paths]
    rejected = [
        {
            'run_name': payload.get('run_name'),
            'reasons': training_contract_reasons(payload, args.hits1_floor),
        }
        for payload in payloads
        if training_contract_reasons(payload, args.hits1_floor)
    ]
    best, complete, eligible = select_joint_payload(payloads, args.hits1_floor)
    if len(complete) < args.minimum_complete:
        raise SystemExit(
            'expected at least {} complete results, found {}'.format(
                args.minimum_complete, len(complete)
            )
        )
    complete_run_names = {payload.get('run_name') for payload in complete}
    manifest_requirements = {}
    for manifest in args.required_run_manifest:
        names = required_run_names(manifest)
        missing = sorted(set(names) - complete_run_names)
        manifest_requirements[str(manifest)] = {
            'required_runs': names,
            'missing_or_invalid_runs': missing,
        }
        if missing:
            raise SystemExit(
                'manifest {} has missing or invalid completed runs: {}'.format(
                    manifest, ', '.join(missing)
                )
            )
    metrics = best['best_joint_test']
    report = {
        'tags': args.tag,
        'complete_runs': len(complete),
        'eligible_runs': len(eligible),
        'rejected_runs': rejected,
        'manifest_requirements': manifest_requirements,
        'selected_run': best['run_name'],
        'selected_metrics': metrics,
        'selected_epoch': best['best_joint_epoch'],
        'joint_checkpoint': best.get('joint_checkpoint'),
        'arguments': best.get('arguments', {}),
        'target_reached': bool(
            best.get('joint_checkpoint')
            and metrics['hits1'] >= args.hits1_floor
            and metrics['hits10'] >= args.hits10_target
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
