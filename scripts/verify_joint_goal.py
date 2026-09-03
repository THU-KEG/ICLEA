#!/usr/bin/env python3
"""Fail-closed verifier for the strict joint ZH_EN Hits@1/Hits@10 goal."""

import argparse
import json
import math
from pathlib import Path


DEFAULT_HITS1 = 0.8887619047619048
DEFAULT_HITS10 = 0.972
EXPECTED_QUERIES = 10500
EXPECTED_CANDIDATES = 19572
EXPECTED_EPOCHS = 300


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
        progress = max(
            0.0,
            float(epoch - step) / max(EXPECTED_EPOCHS - 1 - step, 1),
        )
        return initial + (final - initial) * progress
    return -1.0


def expected_learning_rate(arguments, epoch):
    initial = float(arguments.get('lr', -1.0))
    schedule = arguments.get('lr_schedule', 'step')
    step = int(arguments.get('lr_step_size', 10))
    decay = float(arguments.get('lr_decay', 0.5))
    if schedule == 'constant':
        return initial
    if schedule == 'author-periodic':
        return initial * (0.5 if (epoch + 1) % step == 0 else 1.0)
    if schedule == 'single-step':
        return initial * (decay if epoch >= step else 1.0)
    if schedule == 'cosine':
        minimum = float(arguments.get('lr_min_ratio', 0.1))
        progress = min(max(float(epoch) / max(EXPECTED_EPOCHS - 1, 1), 0.0), 1.0)
        multiplier = minimum + 0.5 * (1.0 - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )
        return initial * multiplier
    return initial * (decay ** (epoch // step))


def expected_temperature(arguments, epoch):
    initial = float(arguments.get('t', -1.0))
    final = float(arguments.get('temperature_final', initial))
    schedule = arguments.get('temperature_schedule', 'constant')
    step = int(arguments.get('temperature_step_epoch', 100))
    if schedule == 'constant':
        return initial
    if schedule == 'step-increase':
        return final if epoch >= step else initial
    progress = min(max(float(epoch) / max(EXPECTED_EPOCHS - 1, 1), 0.0), 1.0)
    return initial + (final - initial) * progress


def expected_reverse_icl_weight(arguments, epoch):
    initial = float(arguments.get('reverse_icl_weight', -1.0))
    final = float(arguments.get('reverse_icl_final_weight', initial))
    schedule = arguments.get('reverse_icl_schedule', 'constant')
    step = int(arguments.get('reverse_icl_step_epoch', 100))
    if schedule == 'constant':
        return initial
    if schedule == 'step-decay':
        return final if epoch >= step else initial
    progress = min(max(float(epoch) / max(EXPECTED_EPOCHS - 1, 1), 0.0), 1.0)
    return initial + (final - initial) * progress


def close_enough(observed, expected, absolute=1e-12):
    return abs(float(observed) - float(expected)) <= max(
        absolute, abs(float(expected)) * 1e-12
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('out/results'))
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--minimum-hits1', type=float, default=DEFAULT_HITS1)
    parser.add_argument('--minimum-hits10', type=float, default=DEFAULT_HITS10)
    parser.add_argument(
        '--checkpoint-evaluation',
        required=True,
        type=Path,
        help='Independent JSON report emitted by scripts/evaluate_checkpoints.py.',
    )
    return parser.parse_args()


def matching_checkpoint_evidence(payload, report):
    checkpoint_value = payload.get('joint_checkpoint')
    selected = payload.get('best_joint_test') or {}
    if not checkpoint_value or not selected:
        return None
    checkpoint_path = Path(checkpoint_value)
    checkpoint = str(checkpoint_path.resolve())
    training_scale = expected_description_scale(
        payload.get('arguments', {}), int(selected.get('epoch', -1))
    )
    if not checkpoint_path.is_file():
        return None
    if (
        report.get('language') != 'zh_en'
        or report.get('setting') != 'original'
        or report.get('evaluation_distance') != 'faiss_squared_l2'
        or report.get('weight_source') != 'online'
        or report.get('runtime_device') != 'cuda'
        or str(report.get('cuda_visible_devices')) not in ('0', '1', '2', '3')
        or int(report.get('visible_cuda_device_count', -1)) != 1
        or report.get('description_scale_override') is not False
        or abs(float(report.get('training_description_scale', -2.0)) - training_scale)
        > 1e-12
        or abs(float(report.get('description_scale', -3.0)) - training_scale) > 1e-12
        or report.get('candidate_scopes') != ['full-target']
        or int(report.get('num_checkpoints', -1)) != 1
    ):
        return None
    rows = report.get('per_checkpoint', ())
    if len(rows) != 1:
        return None
    row = rows[0]
    metrics = row.get('metrics', {}).get('full-target', {})
    valid = (
        str(Path(row.get('checkpoint', '')).resolve()) == checkpoint
        and row.get('weight_source') == 'online'
        and row.get('description_scale_override') is False
        and abs(float(row.get('training_description_scale', -1.0)) - training_scale)
        <= 1e-12
        and abs(float(row.get('description_scale', -2.0)) - training_scale) <= 1e-12
        and int(row.get('selected_epoch', -1)) == int(selected.get('epoch', -2))
        and int(metrics.get('num_queries', -1)) == EXPECTED_QUERIES
        and int(metrics.get('num_candidates', -1)) == EXPECTED_CANDIDATES
        and abs(float(metrics.get('hits1', -1.0)) - float(selected.get('hits1', -2.0)))
        <= 1e-12
        and abs(float(metrics.get('hits10', -1.0)) - float(selected.get('hits10', -2.0)))
        <= 1e-12
    )
    if not valid:
        return None
    return {
        'checkpoint': checkpoint,
        'selected_epoch': int(row['selected_epoch']),
        'metrics': metrics,
        'evaluation_distance': report['evaluation_distance'],
        'weight_source': report['weight_source'],
    }


def audit(path, report, minimum_hits1=DEFAULT_HITS1, minimum_hits10=DEFAULT_HITS10):
    payload = json.loads(path.read_text(encoding='utf-8'))
    arguments = payload.get('arguments', {})
    protocol = payload.get('protocol', {})
    history = payload.get('test_history', [])
    selected = payload.get('best_joint_test') or {}
    eligible = [
        item for item in history if float(item.get('hits1', -1.0)) >= minimum_hits1
    ]
    calculated = (
        max(eligible, key=lambda item: (float(item['hits10']), float(item['hits1'])))
        if eligible else {}
    )
    selection_matches = bool(calculated) and (
        int(selected.get('epoch', -1)) == int(calculated.get('epoch', -2))
        and float(selected.get('hits1', -1.0)) == float(calculated.get('hits1', -2.0))
        and float(selected.get('hits10', -1.0)) == float(calculated.get('hits10', -2.0))
        and int(payload.get('best_joint_epoch', -1)) == int(selected.get('epoch', -2))
    )
    evidence = matching_checkpoint_evidence(payload, report)
    observed_epochs = [int(item.get('epoch', -1)) for item in history]
    description_schedule_valid = (
        arguments.get('description_scale_schedule', 'constant')
        in ('constant', 'linear-increase', 'step-increase')
        and float(arguments.get('description_scale', -1.0)) > 0.0
        and float(
            arguments.get(
                'description_scale_final', arguments.get('description_scale', -1.0)
            )
        ) >= float(arguments.get('description_scale', -1.0))
        and 0 <= int(arguments.get('description_scale_step_epoch', 200)) < EXPECTED_EPOCHS
    )
    lr_schedule_valid = (
        arguments.get('lr_schedule')
        in ('constant', 'author-periodic', 'single-step', 'cosine', 'step')
        and float(arguments.get('lr', -1.0)) > 0.0
        and int(arguments.get('lr_step_size', 0)) >= 1
        and 0.0 < float(arguments.get('lr_decay', 0.0)) <= 1.0
        and 0.0 < float(arguments.get('lr_min_ratio', 0.0)) <= 1.0
    )
    temperature_schedule_valid = (
        arguments.get('temperature_schedule', 'constant')
        in ('constant', 'linear-increase', 'step-increase')
        and float(arguments.get('t', -1.0)) > 0.0
        and float(arguments.get('temperature_final', -1.0)) > 0.0
        and 0 <= int(arguments.get('temperature_step_epoch', 100)) < EXPECTED_EPOCHS
    )
    reverse_schedule_valid = (
        arguments.get('reverse_icl_schedule', 'constant')
        in ('constant', 'linear-decay', 'step-decay')
        and float(arguments.get('reverse_icl_weight', -1.0)) >= 0.0
        and float(arguments.get('reverse_icl_final_weight', -1.0)) >= 0.0
        and 0 <= int(arguments.get('reverse_icl_step_epoch', 100)) < EXPECTED_EPOCHS
    )
    checks = (
        (payload.get('status') == 'complete', 'status is not complete'),
        (payload.get('language') == 'zh_en', 'language is not zh_en'),
        (payload.get('setting') == 'original', 'setting is not original multilingual'),
        (not arguments.get('trans', False), '--trans is enabled'),
        (
            payload.get('profile') in ('paper', 'top1-no-threshold')
            and arguments.get('profile') == payload.get('profile'),
            'profile is not one of the explicit L2 Top-1 profiles',
        ),
        (
            payload.get('rgat_impl') == 'paper-exact-rgat'
            and arguments.get('rgat_impl') == 'paper-exact-rgat'
            and protocol.get('rgat_impl') == 'paper-exact-rgat',
            'RGAT implementation is not paper-exact-rgat',
        ),
        (int(protocol.get('input_slots', -1)) == 16, 'paper RGAT input slot count is not 16'),
        (int(protocol.get('maximum_neighbors', -1)) == 15, 'maximum neighbor count is not 15'),
        (int(protocol.get('gat_layers', -1)) == 1, 'GAT depth is not one hop'),
        (protocol.get('pair_mining') == 'l2', 'pseudo-pair mining is not Faiss L2 Top-1'),
        (
            (
                payload.get('profile') == 'paper'
                and protocol.get('pair_threshold') is not None
                and abs(float(protocol['pair_threshold']) - 1.0) <= 1e-12
            )
            or (
                payload.get('profile') == 'top1-no-threshold'
                and protocol.get('pair_threshold') is None
            ),
            'pseudo-pair threshold is inconsistent with the declared profile',
        ),
        (int(protocol.get('warmup', -1)) == 0, 'ICL warmup is not zero'),
        (protocol.get('negative_set') == 'paper-count', 'negative set is not paper-count'),
        (
            float(protocol.get('description_scale', -1.0)) > 0.0,
            'description scale is absent or non-positive',
        ),
        (
            description_schedule_valid
            and protocol.get('description_scale_schedule', 'constant')
            == arguments.get('description_scale_schedule', 'constant')
            and abs(
                float(protocol.get('description_scale', -1.0))
                - float(arguments.get('description_scale', -2.0))
            ) <= 1e-12
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
            == int(arguments.get('description_scale_step_epoch', 200)),
            'description scale schedule metadata is invalid or inconsistent',
        ),
        (
            lr_schedule_valid
            and protocol.get('lr_schedule_name') == arguments.get('lr_schedule')
            and close_enough(protocol.get('lr_initial', -1.0), arguments.get('lr', -2.0))
            and close_enough(
                protocol.get('lr_min_ratio', -1.0),
                arguments.get('lr_min_ratio', -2.0),
            )
            and int(protocol.get('lr_step_size', -1))
            == int(arguments.get('lr_step_size', -2))
            and close_enough(
                protocol.get('lr_decay', -1.0), arguments.get('lr_decay', -2.0)
            ),
            'learning-rate schedule metadata is invalid or inconsistent',
        ),
        (
            temperature_schedule_valid
            and protocol.get('temperature_schedule')
            == arguments.get('temperature_schedule')
            and close_enough(
                protocol.get('temperature_initial', -1.0), arguments.get('t', -2.0)
            )
            and close_enough(
                protocol.get('temperature_final', -1.0),
                arguments.get('temperature_final', -2.0),
            )
            and int(protocol.get('temperature_step_epoch', -1))
            == int(arguments.get('temperature_step_epoch', -2)),
            'temperature schedule metadata is invalid or inconsistent',
        ),
        (
            reverse_schedule_valid
            and close_enough(
                protocol.get('reverse_icl_weight', -1.0),
                arguments.get('reverse_icl_weight', -2.0),
            )
            and protocol.get('reverse_icl_schedule')
            == arguments.get('reverse_icl_schedule')
            and close_enough(
                protocol.get('reverse_icl_final_weight', -1.0),
                arguments.get('reverse_icl_final_weight', -2.0),
            )
            and int(protocol.get('reverse_icl_step_epoch', -1))
            == int(arguments.get('reverse_icl_step_epoch', -2)),
            'reverse-ICL schedule metadata is invalid or inconsistent',
        ),
        (
            protocol.get('momentum_init') == 'copy-online',
            'momentum encoder was not initialized from the online encoder',
        ),
        (int(arguments.get('epoch', -1)) == EXPECTED_EPOCHS, 'configured epoch is not 300'),
        (len(history) == EXPECTED_EPOCHS, 'test history does not contain 300 epochs'),
        (
            observed_epochs == list(range(EXPECTED_EPOCHS)),
            'test history epoch sequence is not exactly 0 through 299',
        ),
        (
            description_schedule_valid
            and all(
                abs(
                    float(item.get('description_scale', -1.0))
                    - expected_description_scale(arguments, int(item.get('epoch', -1)))
                ) <= 1e-12
                for item in history
            ),
            'test history description scales do not match the declared schedule',
        ),
        (
            lr_schedule_valid
            and all(
                close_enough(
                    item.get('learning_rate', -1.0),
                    expected_learning_rate(arguments, int(item.get('epoch', -1))),
                    absolute=1e-18,
                )
                for item in history
            ),
            'test history learning rates do not match the declared schedule',
        ),
        (
            temperature_schedule_valid
            and all(
                close_enough(
                    item.get('temperature', -1.0),
                    expected_temperature(arguments, int(item.get('epoch', -1))),
                )
                for item in history
            ),
            'test history temperatures do not match the declared schedule',
        ),
        (
            reverse_schedule_valid
            and all(
                close_enough(
                    item.get('reverse_icl_weight', -1.0),
                    expected_reverse_icl_weight(
                        arguments, int(item.get('epoch', -1))
                    ),
                )
                for item in history
            ),
            'test history reverse-ICL weights do not match the declared schedule',
        ),
        (int(payload.get('trained_epochs', -1)) == EXPECTED_EPOCHS, 'trained_epochs is not 300'),
        (payload.get('stopped_early') is False, 'run stopped early'),
        (
            protocol.get('test_candidates') == 'complete_target_kg',
            'candidate scope is not the complete target KG',
        ),
        (
            protocol.get('evaluation_distance') == 'faiss_squared_l2',
            'training evaluation distance is not Faiss squared-L2',
        ),
        (protocol.get('validation_access') == 'none', 'validation data was accessed'),
        (
            str(protocol.get('cuda_visible_devices')) in ('0', '1', '2', '3')
            and int(protocol.get('visible_cuda_device_count', -1)) == 1
            and int(protocol.get('physical_gpu', -1))
            == int(protocol.get('cuda_visible_devices', -2)),
            'training was not scoped to exactly one physical GPU 0-3',
        ),
        (protocol.get('selection_protocol') == 'test_best', 'selection is not test-best'),
        (
            int(protocol.get('test_evaluations', -1)) == EXPECTED_EPOCHS,
            'test evaluation count is not 300',
        ),
        (
            all(int(item.get('num_queries', -1)) == EXPECTED_QUERIES for item in history),
            'one or more evaluations do not contain 10500 test queries',
        ),
        (
            arguments.get('joint_hits1_floor') is not None
            and abs(float(arguments['joint_hits1_floor']) - minimum_hits1) <= 1e-12,
            'configured joint Hits@1 floor does not match the verifier floor',
        ),
        (selection_matches, 'best_joint_test is not the constrained history optimum'),
        (float(selected.get('hits1', -1.0)) >= minimum_hits1, 'joint checkpoint Hits@1 is below target'),
        (float(selected.get('hits10', -1.0)) >= minimum_hits10, 'joint checkpoint Hits@10 is below target'),
        (
            evidence is not None,
            'joint checkpoint lacks one matching independent online full-target Faiss evaluation',
        ),
    )
    failures = [message for passed, message in checks if not passed]
    return payload, failures, evidence


def main():
    args = parse_args()
    result_path = args.results_dir / (args.run_name + '.json')
    if not result_path.is_file():
        raise SystemExit('result JSON is missing: {}'.format(result_path))
    report = json.loads(args.checkpoint_evaluation.read_text(encoding='utf-8'))
    payload, failures, evidence = audit(
        result_path,
        report,
        minimum_hits1=args.minimum_hits1,
        minimum_hits10=args.minimum_hits10,
    )
    rendered = {
        'run_name': payload.get('run_name'),
        'minimum_hits1': args.minimum_hits1,
        'minimum_hits10': args.minimum_hits10,
        'selected': payload.get('best_joint_test'),
        'failures': failures,
        'independent_checkpoint_evidence': evidence,
        'strict_contract': {
            'language': 'zh_en',
            'setting': 'original multilingual',
            'epochs': EXPECTED_EPOCHS,
            'selection': 'highest Hits@10 subject to the Hits@1 floor',
            'validation_access': 'none',
            'candidate_scope': 'complete target KG',
            'target_candidates': EXPECTED_CANDIDATES,
            'test_queries': EXPECTED_QUERIES,
            'distance': 'Faiss squared L2',
            'checkpoint_weights': 'online',
            'profile': 'paper threshold 1.0 or explicit top1-no-threshold',
            'rgat': 'paper-exact-rgat with center plus 15 neighbors',
            'pseudo_pairs': 'bidirectional Faiss L2 Top-1 with profile-consistent threshold',
            'negative_set': 'paper-count',
            'momentum_initialization': 'copy-online',
            'training_gpu_scope': 'exactly one physical GPU in 0-3',
            'independent_evaluation_gpu_scope': 'exactly one physical GPU in 0-3',
        },
    }
    print(json.dumps(rendered, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
