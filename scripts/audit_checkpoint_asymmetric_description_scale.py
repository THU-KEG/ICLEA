#!/usr/bin/env python3
"""Diagnostic full-target grid for side-specific description input scaling."""

import argparse
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.ICLEA import MyEmbedder, Trainer, VOCAB_SIZE  # noqa: E402
from scripts.evaluate_checkpoints import hits, trainer_argv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--scales', default='1.5,2.0,2.25,2.5,3.0')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def encode_side(trainer, dataset, scale):
    trainer.args.description_scale = float(scale)
    trainer.model.args.description_scale = float(scale)
    trainer.model.description_scale = float(scale)
    with torch.no_grad():
        trainer.model.eval()
        return trainer._encode_dataset(dataset)


def main():
    cli = parse_args()
    if cli.device == 'cpu' and torch.cuda.is_available():
        raise RuntimeError('CPU diagnostics require CUDA_VISIBLE_DEVICES=""')
    payload = torch.load(str(cli.checkpoint.resolve()), map_location='cpu')
    original_argv = sys.argv
    sys.argv = trainer_argv(payload['arguments'])
    try:
        trainer = Trainer(training=False)
    finally:
        sys.argv = original_argv
    trainer.device = torch.device(cli.device)
    trainer.model = MyEmbedder(trainer.args, VOCAB_SIZE).to(trainer.device)
    trainer.model.load_state_dict(payload['online_model'])
    scales = [float(value) for value in cli.scales.split(',')]
    if any(value <= 0.0 for value in scales):
        raise ValueError('all scales must be positive')
    vectors_1 = {
        str(scale): encode_side(trainer, trainer.dataset1, scale) for scale in scales
    }
    vectors_2 = {
        str(scale): encode_side(trainer, trainer.dataset2, scale) for scale in scales
    }
    source, target = trainer._link_indices(
        trainer.dataset1, trainer.dataset2, trainer.test_link_tensor
    )
    grid = {}
    for source_scale in scales:
        for target_scale in scales:
            key = '{}:{}'.format(source_scale, target_scale)
            grid[key] = hits(
                vectors_1[str(source_scale)][source],
                vectors_2[str(target_scale)],
                target,
            )
    report = {
        'diagnostic_only': True,
        'checkpoint': str(cli.checkpoint.resolve()),
        'selected_epoch': int(payload['epoch']),
        'weight_source': 'online',
        'candidate_scope': 'full-target',
        'evaluation_distance': 'faiss_squared_l2',
        'runtime_device': cli.device,
        'source_scales': scales,
        'target_scales': scales,
        'grid': grid,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    best = max(
        grid.items(),
        key=lambda item: (item[1]['hits10'], item[1]['hits1']),
    )
    print('BEST_ASYMMETRIC_DESCRIPTION_SCALE=' + json.dumps({
        'scales': best[0],
        'metrics': best[1],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
