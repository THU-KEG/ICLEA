#!/usr/bin/env python3
"""Sweep transparent weighted embedding means for checkpoint diagnostics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.ICLEA import MyEmbedder, Trainer, VOCAB_SIZE  # noqa: E402
from scripts.evaluate_checkpoints import (  # noqa: E402
    encode_checkpoint,
    hits,
    normalize_rows,
    trainer_argv,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-checkpoint', required=True)
    parser.add_argument('--auxiliary-checkpoint', action='append', required=True)
    parser.add_argument('--reference-weight-start', type=float, default=0.5)
    parser.add_argument('--reference-weight-stop', type=float, default=1.0)
    parser.add_argument('--reference-weight-step', type=float, default=0.01)
    parser.add_argument('--hits1-floor', type=float, default=0.0)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    paths = [Path(cli.reference_checkpoint).resolve()] + [
        Path(value).resolve() for value in cli.auxiliary_checkpoint
    ]
    payloads = [torch.load(str(path), map_location='cpu') for path in paths]
    arguments = payloads[0]['arguments']
    for payload in payloads[1:]:
        other = payload['arguments']
        for key in ('language', 'model_language', 'trans', 'profile', 'combine', 'rgat_impl'):
            if other.get(key) != arguments.get(key):
                raise ValueError('Checkpoint argument mismatch for {}'.format(key))

    original_argv = sys.argv
    sys.argv = trainer_argv(arguments)
    try:
        trainer = Trainer(training=False)
    finally:
        sys.argv = original_argv
    trainer.device = torch.device(cli.device)
    trainer.model = MyEmbedder(trainer.args, VOCAB_SIZE).to(trainer.device)

    vectors = [encode_checkpoint(trainer, payload, 'online') for payload in payloads]
    source, target = trainer._link_indices(
        trainer.dataset1, trainer.dataset2, trainer.test_link_tensor
    )
    weights = np.arange(
        cli.reference_weight_start,
        cli.reference_weight_stop + cli.reference_weight_step / 2.0,
        cli.reference_weight_step,
    )
    sweep = []
    auxiliary_count = len(vectors) - 1
    for reference_weight in weights:
        auxiliary_weight = (1.0 - float(reference_weight)) / auxiliary_count
        vector_1 = reference_weight * vectors[0][0]
        vector_2 = reference_weight * vectors[0][1]
        for auxiliary_1, auxiliary_2 in vectors[1:]:
            vector_1 = vector_1 + auxiliary_weight * auxiliary_1
            vector_2 = vector_2 + auxiliary_weight * auxiliary_2
        vector_1 = normalize_rows(vector_1)
        vector_2 = normalize_rows(vector_2)
        metrics = hits(vector_1[source], vector_2, target)
        sweep.append({
            'reference_weight': float(round(reference_weight, 10)),
            'auxiliary_weight_each': float(auxiliary_weight),
            **metrics,
        })

    eligible = [row for row in sweep if row['hits1'] >= cli.hits1_floor]
    best_eligible = max(
        eligible,
        key=lambda row: (row['hits10'], row['hits1']),
    ) if eligible else None
    report = {
        'reference_checkpoint': str(paths[0]),
        'auxiliary_checkpoints': [str(path) for path in paths[1:]],
        'candidate_scope': 'full-target',
        'evaluation_distance': 'faiss_squared_l2',
        'hits1_floor': cli.hits1_floor,
        'best_eligible': best_eligible,
        'sweep': sweep,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = cli.output.with_suffix(cli.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temporary.replace(cli.output)
    print('WEIGHTED_ENSEMBLE_AUDIT_JSON=' + json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
