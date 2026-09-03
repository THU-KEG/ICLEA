#!/usr/bin/env python3
"""Evaluate one or more ICLEA checkpoints under explicit candidate scopes."""

import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.ICLEA import (  # noqa: E402
    MyEmbedder,
    Trainer,
    VOCAB_SIZE,
    description_scale_for_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument(
        '--candidate-scope',
        choices=('full-target', 'test-subset', 'both'),
        default='both',
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--weight-source',
        choices=('online', 'momentum', 'online-momentum-ensemble'),
        default='online',
        help='Select the saved encoder weights used for retrieval evaluation.',
    )
    parser.add_argument(
        '--device',
        choices=('cuda', 'cpu'),
        default='cuda',
        help='Runtime used only for independent checkpoint encoding.',
    )
    parser.add_argument(
        '--description-scale-override',
        type=float,
        help=(
            'Diagnostic-only input perturbation. Omit for an independent strict '
            'reproduction of the checkpoint training configuration.'
        ),
    )
    return parser.parse_args()


def trainer_argv(arguments):
    values = [
        'evaluate_checkpoints.py',
        '--language', arguments['language'],
        '--model_language', arguments['model_language'],
        '--profile', arguments.get('profile', 'paper'),
        '--rgat_impl', arguments.get('rgat_impl', 'author-code'),
        '--seed', str(arguments.get('seed', 37)),
        '--batch_size', str(arguments.get('batch_size', 64)),
        '--description_scale', str(arguments.get('description_scale', 1.0)),
        '--description_scale_schedule', arguments.get('description_scale_schedule', 'constant'),
        '--description_scale_final', str(arguments.get(
            'description_scale_final', arguments.get('description_scale', 1.0)
        )),
        '--description_scale_step_epoch', str(arguments.get('description_scale_step_epoch', 200)),
        '--gat_num', str(arguments.get('gat_num', 1)),
        '--t', str(arguments.get('t', 0.08)),
        '--temperature_schedule', arguments.get('temperature_schedule', 'constant'),
        '--temperature_final', str(arguments.get('temperature_final', arguments.get('t', 0.08))),
        '--temperature_step_epoch', str(arguments.get('temperature_step_epoch', 100)),
        '--momentum', str(arguments.get('momentum', 0.9999)),
        '--momentum_init', arguments.get('momentum_init', 'copy-online'),
        '--pair_mining', arguments.get('pair_mining', 'l2'),
        '--pair_csls_k', str(arguments.get('pair_csls_k', 10)),
        '--pair_candidate_pool', str(arguments.get('pair_candidate_pool', 100)),
        '--lr', str(arguments.get('lr', 1e-6)),
        '--dropout', str(arguments.get('dropout', 0.3)),
        '--reverse_icl_weight', str(arguments.get('reverse_icl_weight', 0.0)),
        '--reverse_icl_schedule', arguments.get('reverse_icl_schedule', 'constant'),
        '--reverse_icl_final_weight', str(arguments.get(
            'reverse_icl_final_weight', arguments.get('reverse_icl_weight', 0.0)
        )),
        '--reverse_icl_step_epoch', str(arguments.get('reverse_icl_step_epoch', 100)),
        '--icl_beta', str(arguments.get('icl_beta', 0.9)),
        '--icl_source_inbatch', arguments.get('icl_source_inbatch', 'pseudo-target'),
        '--eval_batch_size', str(arguments.get('eval_batch_size', 256)),
        '--lr_schedule', arguments.get('lr_schedule', 'step'),
        '--lr_step_size', str(arguments.get('lr_step_size', 10)),
        '--lr_decay', str(arguments.get('lr_decay', 0.5)),
        '--lr_min_ratio', str(arguments.get('lr_min_ratio', 0.1)),
        '--batch_order', arguments.get('batch_order', 'reshuffle'),
        '--negative_set', arguments.get('negative_set', 'queue-only'),
    ]
    if arguments.get('trans'):
        values.append('--trans')
    if arguments.get('center_norm'):
        values.append('--center_norm')
    if not arguments.get('neighbor_norm', True):
        values.append('--no_neighbor_norm')
    if not arguments.get('emb_norm', True):
        values.append('--no_emb_norm')
    if not arguments.get('combine', True):
        values.append('--no_combine')
    if arguments.get('exclude_false_negatives'):
        values.append('--exclude_false_negatives')
    return values


def normalize_rows(vectors):
    denominator = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(denominator, 1e-12)


def hits(queries, candidates, target):
    queries = np.ascontiguousarray(queries.astype(np.float32))
    candidates = np.ascontiguousarray(candidates.astype(np.float32))
    index = faiss.IndexFlatL2(candidates.shape[1])
    index.add(candidates)
    _, neighbors = index.search(queries, 10)
    return {
        'num_queries': int(len(target)),
        'num_candidates': int(len(candidates)),
        'hits1': float((neighbors[:, 0] == target).astype(np.int32).mean()),
        'hits10': float((neighbors == target[:, np.newaxis]).astype(np.int32).any(axis=1).mean()),
    }


def evaluate_scopes(trainer, vector_1, vector_2, scopes):
    source, target = trainer._link_indices(
        trainer.dataset1, trainer.dataset2, trainer.test_link_tensor
    )
    metrics = {}
    if 'full-target' in scopes:
        metrics['full-target'] = hits(vector_1[source], vector_2, target)
    if 'test-subset' in scopes:
        subset_target = np.arange(len(target), dtype=np.int64)
        metrics['test-subset'] = hits(
            vector_1[source], vector_2[target], subset_target
        )
    return metrics


def encode_checkpoint(trainer, payload, weight_source):
    if weight_source in ('online', 'momentum'):
        key = 'online_model' if weight_source == 'online' else 'momentum_model'
        trainer.model.load_state_dict(payload[key])
        return trainer._encode_both()

    trainer.model.load_state_dict(payload['online_model'])
    online_1, online_2 = trainer._encode_both()
    trainer.model.load_state_dict(payload['momentum_model'])
    momentum_1, momentum_2 = trainer._encode_both()
    return (
        normalize_rows((online_1 + momentum_1) / 2.0),
        normalize_rows((online_2 + momentum_2) / 2.0),
    )


def main():
    cli = parse_args()
    checkpoints = [Path(value).resolve() for value in cli.checkpoint]
    payloads = [torch.load(str(path), map_location='cpu') for path in checkpoints]
    arguments = payloads[0]['arguments']
    for payload in payloads[1:]:
        other = payload['arguments']
        for key in (
            'language',
            'model_language',
            'trans',
            'profile',
            'combine',
            'rgat_impl',
            'neighbor_size',
        ):
            if other.get(key) != arguments.get(key):
                raise ValueError('Checkpoint argument mismatch for {}'.format(key))

    if cli.device == 'cpu' and torch.cuda.is_available():
        raise RuntimeError(
            'CPU audit must run with CUDA_VISIBLE_DEVICES="" so it cannot initialize '
            'CUDA contexts on reserved GPUs.'
        )
    original_argv = sys.argv
    sys.argv = trainer_argv(arguments)
    try:
        trainer = Trainer(training=False)
    finally:
        sys.argv = original_argv
    if cli.description_scale_override is not None:
        if cli.description_scale_override <= 0.0:
            raise ValueError('--description-scale-override must be positive')
    trainer.device = torch.device(cli.device)
    trainer.model = MyEmbedder(trainer.args, VOCAB_SIZE).to(trainer.device)
    scopes = (
        ('full-target', 'test-subset')
        if cli.candidate_scope == 'both'
        else (cli.candidate_scope,)
    )

    vector_1_sum = None
    vector_2_sum = None
    per_checkpoint = []
    for path, payload in zip(checkpoints, payloads):
        configured_scale = description_scale_for_epoch(
            trainer.args, int(payload['epoch'])
        )
        effective_scale = (
            float(cli.description_scale_override)
            if cli.description_scale_override is not None
            else configured_scale
        )
        trainer.model.description_scale = effective_scale
        vector_1, vector_2 = encode_checkpoint(trainer, payload, cli.weight_source)
        vector_1_sum = vector_1 if vector_1_sum is None else vector_1_sum + vector_1
        vector_2_sum = vector_2 if vector_2_sum is None else vector_2_sum + vector_2
        per_checkpoint.append({
            'checkpoint': str(path),
            'selected_epoch': int(payload['epoch']),
            'weight_source': cli.weight_source,
            'training_description_scale': configured_scale,
            'description_scale': effective_scale,
            'description_scale_override': cli.description_scale_override is not None,
            'metrics': evaluate_scopes(trainer, vector_1, vector_2, scopes),
        })

    count = float(len(checkpoints))
    ensemble_1 = normalize_rows(vector_1_sum / count)
    ensemble_2 = normalize_rows(vector_2_sum / count)
    report = {
        'language': arguments['language'],
        'setting': 'translated' if arguments.get('trans') else 'original',
        'num_checkpoints': len(checkpoints),
        'candidate_scopes': list(scopes),
        'evaluation_distance': 'faiss_squared_l2',
        'runtime_device': cli.device,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'visible_cuda_device_count': (
            torch.cuda.device_count() if cli.device == 'cuda' else 0
        ),
        'training_description_scale': per_checkpoint[0]['training_description_scale'],
        'description_scale': per_checkpoint[0]['description_scale'],
        'description_scale_override': cli.description_scale_override is not None,
        'weight_source': cli.weight_source,
        'per_checkpoint': per_checkpoint,
        'normalized_embedding_mean_ensemble': evaluate_scopes(
            trainer, ensemble_1, ensemble_2, scopes
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = cli.output.with_suffix(cli.output.suffix + '.tmp')
        temporary.write_text(rendered + '\n', encoding='utf-8')
        temporary.replace(cli.output)
    print('CHECKPOINT_EVALUATION_JSON=' + json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
