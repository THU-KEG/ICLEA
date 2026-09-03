#!/usr/bin/env python3
"""Fail-closed comparison of two ICLEA checkpoint state dictionaries."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument(
        '--weight-source', choices=('online', 'momentum'), default='online'
    )
    parser.add_argument('--maximum-absolute-delta', type=float, required=True)
    parser.add_argument('--require-before-epoch', type=int)
    parser.add_argument('--require-after-epoch', type=int)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compare_checkpoints(before_path, after_path, weight_source):
    before_payload = torch.load(str(before_path), map_location='cpu')
    after_payload = torch.load(str(after_path), map_location='cpu')
    state_key = '{}_model'.format(weight_source)
    if state_key not in before_payload or state_key not in after_payload:
        raise ValueError('both checkpoints must contain {}'.format(state_key))
    before_state = before_payload[state_key]
    after_state = after_payload[state_key]
    if set(before_state) != set(after_state):
        missing = sorted(set(before_state) - set(after_state))
        extra = sorted(set(after_state) - set(before_state))
        raise ValueError(
            'state_dict key mismatch: missing={} extra={}'.format(missing, extra)
        )

    maximum = 0.0
    changed_tensors = 0
    changed_elements = 0
    tensor_rows = []
    for name in sorted(before_state):
        before = before_state[name]
        after = after_state[name]
        if not torch.is_tensor(before) or not torch.is_tensor(after):
            raise TypeError('state_dict entry {} is not a tensor'.format(name))
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(
                'tensor metadata mismatch for {}: {} {} versus {} {}'.format(
                    name, tuple(before.shape), before.dtype,
                    tuple(after.shape), after.dtype,
                )
            )
        unequal = before.ne(after)
        count = int(unequal.sum().item())
        if torch.is_floating_point(before):
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                raise ValueError('non-finite tensor found in {}'.format(name))
            delta = float(
                (before.to(torch.float64) - after.to(torch.float64))
                .abs().max().item()
            ) if before.numel() else 0.0
        else:
            delta = 1.0 if count else 0.0
        if count:
            changed_tensors += 1
            changed_elements += count
            tensor_rows.append({
                'name': name,
                'changed_elements': count,
                'maximum_absolute_delta': delta,
            })
        maximum = max(maximum, delta)

    return {
        'before_epoch': int(before_payload.get('epoch', -1)),
        'after_epoch': int(after_payload.get('epoch', -1)),
        'weight_source': weight_source,
        'tensor_count': len(before_state),
        'changed_tensor_count': changed_tensors,
        'changed_element_count': changed_elements,
        'maximum_absolute_delta': maximum,
        'changed_tensors': tensor_rows,
    }


def main():
    args = parse_args()
    if args.maximum_absolute_delta < 0.0:
        raise ValueError('--maximum-absolute-delta must be non-negative')
    before = args.before.resolve()
    after = args.after.resolve()
    if not before.is_file() or not after.is_file():
        raise FileNotFoundError('both checkpoint paths must exist')
    report = compare_checkpoints(before, after, args.weight_source)
    if (
        args.require_before_epoch is not None
        and report['before_epoch'] != args.require_before_epoch
    ):
        raise ValueError('unexpected before checkpoint epoch')
    if (
        args.require_after_epoch is not None
        and report['after_epoch'] != args.require_after_epoch
    ):
        raise ValueError('unexpected after checkpoint epoch')
    report.update({
        'before_checkpoint': str(before),
        'after_checkpoint': str(after),
        'before_sha256': sha256(before),
        'after_sha256': sha256(after),
        'maximum_allowed_absolute_delta': args.maximum_absolute_delta,
        'passed': report['maximum_absolute_delta'] <= args.maximum_absolute_delta,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    os.replace(str(temporary), str(args.output))
    if not report['passed']:
        raise SystemExit(
            'checkpoint drift {:.12g} exceeds {:.12g}'.format(
                report['maximum_absolute_delta'],
                args.maximum_absolute_delta,
            )
        )


if __name__ == '__main__':
    main()
