# coding: UTF-8
import os
import argparse
import json
import math
import pickle
import random
import time
from datetime import datetime
from os.path import join

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.nn import Parameter
from tqdm import tqdm

from settings import (
    BATCH_SIZE,
    DATA_DIR,
    DESC_DIM,
    LaBSE_DIM,
    MULTI_HEAD_DIM,
    NEIGHBOR_SIZE,
    PROJ_DIR,
    VOCAB_SIZE,
    fix_seed,
)
from loader.DBP15KRawDataset_ICLEA import DBP15KRawDataset
from script.preprocess.deal_raw_dataset_ICLEA import MyRawdataset


AUTHOR_RGAT = 'author-code'
PAPER_EXACT_RGAT = 'paper-exact-rgat'


def parse_options(parser):
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--language', choices=('zh_en', 'ja_en', 'fr_en'), default='zh_en')
    parser.add_argument('--model_language', type=str, default='zh_en')
    parser.add_argument('--model', type=str, default='LaBSE')
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument(
        '--profile',
        choices=('paper', 'top1-no-threshold'),
        default='paper',
        help='paper uses L2<1.0 and ICL from epoch 0; top1-no-threshold keeps every Faiss Top-1 pair.',
    )

    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--queue_length', type=int, default=32)
    parser.add_argument(
        '--description_scale',
        type=float,
        default=1.0,
        help=(
            'Scale the normalized description half before neighborhood aggregation. '
            'The paper/default concatenation uses 1.0.'
        ),
    )
    parser.add_argument(
        '--description_scale_schedule',
        choices=('constant', 'linear-increase', 'step-increase'),
        default='constant',
        help='Optionally increase description input weight later without changing the model or loss.',
    )
    parser.add_argument('--description_scale_final', type=float, default=None)
    parser.add_argument('--description_scale_step_epoch', type=int, default=200)

    parser.add_argument('--center_norm', action='store_true')
    parser.add_argument('--no_neighbor_norm', action='store_false', dest='neighbor_norm')
    parser.add_argument('--no_emb_norm', action='store_false', dest='emb_norm')
    parser.add_argument('--no_combine', action='store_false', dest='combine')
    parser.add_argument('--trans', action='store_true')
    parser.set_defaults(neighbor_norm=True, emb_norm=True, combine=True)

    parser.add_argument('--gat_num', type=int, default=1)
    parser.add_argument(
        '--rgat_impl',
        choices=(AUTHOR_RGAT, PAPER_EXACT_RGAT),
        default=AUTHOR_RGAT,
        help=(
            'author-code preserves the released one-layer implementation and 15 total slots; '
            'paper-exact-rgat restores the paper equations and uses a center plus 15 neighbors.'
        ),
    )

    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument(
        '--temperature_schedule',
        choices=('constant', 'linear-increase', 'step-increase'),
        default='constant',
        help=(
            'Keep the NCE/ICL equations unchanged while optionally increasing their '
            'temperature late in training to reduce hard-negative over-concentration.'
        ),
    )
    parser.add_argument(
        '--temperature_final',
        type=float,
        default=None,
        help='Final temperature for a non-constant temperature schedule; defaults to --t.',
    )
    parser.add_argument(
        '--temperature_step_epoch',
        type=int,
        default=100,
        help='Zero-based switch epoch for --temperature_schedule step-increase.',
    )
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument(
        '--momentum_init',
        choices=('copy-online', 'independent'),
        default='copy-online',
        help=(
            'copy-online follows the paper pseudocode theta_prime <- theta. '
            'independent reconstructs the released source behavior for a controlled audit.'
        ),
    )
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument(
        '--reverse_icl_weight',
        type=float,
        default=0.0,
        help=(
            'Optional weight for the extra reverse-anchor ICL term present in the '
            'released source but disabled there by a variable-name bug. The paper '
            'equations and clean default use 0.'
        ),
    )
    parser.add_argument(
        '--reverse_icl_schedule',
        choices=('constant', 'linear-decay', 'step-decay'),
        default='constant',
        help='Schedule only the weight of the optional released-source reverse ICL term.',
    )
    parser.add_argument(
        '--reverse_icl_final_weight',
        type=float,
        default=None,
        help='Final reverse-ICL weight for a decay schedule; defaults to the initial weight.',
    )
    parser.add_argument(
        '--reverse_icl_step_epoch',
        type=int,
        default=100,
        help='Zero-based switch epoch for --reverse_icl_schedule step-decay.',
    )
    parser.add_argument(
        '--icl_beta',
        type=float,
        default=0.9,
        help=(
            'Weight on source-KG negatives in the interactive contrastive loss; '
            'the paper reports beta=0.9.'
        ),
    )
    parser.add_argument(
        '--icl_source_inbatch',
        choices=('pseudo-target', 'source'),
        default='pseudo-target',
        help=(
            'Select the B-1 in-batch negatives for the beta-weighted source-KG ICL '
            'term. pseudo-target preserves the released/previous clean behavior; '
            'source follows Equation 8 and the paper-count statement.'
        ),
    )
    parser.add_argument('--warmup', default=None, type=int)
    parser.add_argument('--pair_threshold', type=float, default=None)
    parser.add_argument(
        '--pair_mining',
        choices=('l2', 'csls'),
        default='l2',
        help='Pseudo-pair retrieval rule. l2 is the paper default; csls is an opt-in hubness audit.',
    )
    parser.add_argument('--pair_csls_k', type=int, default=10)
    parser.add_argument('--pair_candidate_pool', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--eval_every_epochs', type=int, default=1)
    parser.add_argument(
        '--joint_hits1_floor',
        type=float,
        default=None,
        help=(
            'Optionally save the checkpoint with the highest Hits@10 among epochs whose '
            'Hits@1 reaches this floor. This changes checkpoint retention only, not training.'
        ),
    )
    parser.add_argument(
        '--snapshot_epochs',
        type=str,
        default='',
        help=(
            'Comma-separated zero-based epochs to archive for diagnostic comparisons. '
            'This changes checkpoint retention only, not training.'
        ),
    )
    parser.add_argument(
        '--selection_protocol',
        choices=('test-best',),
        default='test-best',
        help='Train every epoch and report the highest monitored test Hits@1.',
    )
    parser.add_argument(
        '--lr_schedule',
        choices=('step', 'single-step', 'constant', 'author-periodic', 'cosine'),
        default='step',
        help=(
            'author-periodic reproduces the released implementation: use half the base LR '
            'only on every tenth epoch, then return to the base LR.'
        ),
    )
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--lr_min_ratio', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default=join(PROJ_DIR, 'out'))
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--feature_device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument(
        '--batch_order',
        choices=('reshuffle', 'fixed'),
        default='reshuffle',
        help='fixed reproduces the released code, which shuffled batches once before training.',
    )
    parser.add_argument(
        '--negative_set',
        choices=('queue-only', 'paper-count'),
        default='queue-only',
        help=(
            'paper-count adds the other B-1 samples in the positive batch, matching the '
            'paper statement that each anchor has (L+1)*B-1 negatives.'
        ),
    )
    parser.add_argument(
        '--exclude_false_negatives',
        action='store_true',
        help=(
            'Mask queue rows and in-batch keys whose entity ID equals the matching '
            'query/positive entity in that KG. This is opt-in so historical results '
            'remain reconstructible.'
        ),
    )
    parser.add_argument('--no_fused_momentum', action='store_false', dest='fused_momentum')
    parser.add_argument('--max_steps', type=int, default=0)
    parser.set_defaults(fused_momentum=True)

    args = parser.parse_args()
    if args.warmup is None:
        args.warmup = 0
    elif args.warmup != 0:
        parser.error('{} profile fixes --warmup=0'.format(args.profile))
    if args.profile == 'paper':
        if args.pair_threshold is not None and args.pair_threshold != 1.0:
            parser.error('paper profile fixes --pair_threshold=1.0')
        args.pair_threshold = 1.0
    elif args.pair_threshold is not None:
        parser.error('top1-no-threshold profile does not accept --pair_threshold')
    if args.eval_every_epochs < 1:
        parser.error('--eval_every_epochs must be at least 1')
    if args.t <= 0.0:
        parser.error('--t must be positive')
    if args.description_scale <= 0.0:
        parser.error('--description_scale must be positive')
    if args.description_scale_final is None:
        args.description_scale_final = args.description_scale
    if args.description_scale_final <= 0.0:
        parser.error('--description_scale_final must be positive')
    if (
        args.description_scale_schedule != 'constant'
        and args.description_scale_final < args.description_scale
    ):
        parser.error('non-constant description schedules require final >= initial')
    if (
        args.description_scale_schedule == 'constant'
        and args.description_scale_final != args.description_scale
    ):
        parser.error('constant description schedule requires equal initial/final scales')
    if not 0 <= args.description_scale_step_epoch < args.epoch:
        parser.error('--description_scale_step_epoch must be in [0, epoch)')
    if args.temperature_final is None:
        args.temperature_final = args.t
    if args.temperature_final <= 0.0:
        parser.error('--temperature_final must be positive')
    if args.temperature_schedule != 'constant' and args.temperature_final < args.t:
        parser.error('non-constant temperature schedules require temperature_final >= t')
    if args.temperature_schedule == 'constant' and args.temperature_final != args.t:
        parser.error('constant temperature schedule requires temperature_final == t')
    if not 0 <= args.temperature_step_epoch < args.epoch:
        parser.error('--temperature_step_epoch must be in [0, epoch)')
    if args.joint_hits1_floor is not None and not 0.0 <= args.joint_hits1_floor <= 1.0:
        parser.error('--joint_hits1_floor must be in [0, 1]')
    try:
        args.snapshot_epochs = sorted({
            int(value.strip())
            for value in args.snapshot_epochs.split(',')
            if value.strip()
        })
    except ValueError:
        parser.error('--snapshot_epochs must be a comma-separated list of integers')
    if any(epoch < 0 or epoch >= args.epoch for epoch in args.snapshot_epochs):
        parser.error('--snapshot_epochs values must be in [0, epoch)')
    if args.pair_csls_k < 1:
        parser.error('--pair_csls_k must be at least 1')
    if args.pair_candidate_pool < args.pair_csls_k:
        parser.error('--pair_candidate_pool must be no smaller than --pair_csls_k')
    if args.lr_step_size < 1:
        parser.error('--lr_step_size must be at least 1')
    if not 0.0 < args.lr_decay <= 1.0:
        parser.error('--lr_decay must be in (0, 1]')
    if not 0.0 <= args.lr_min_ratio <= 1.0:
        parser.error('--lr_min_ratio must be in [0, 1]')
    if args.reverse_icl_weight < 0.0:
        parser.error('--reverse_icl_weight must be non-negative')
    if args.reverse_icl_final_weight is None:
        args.reverse_icl_final_weight = args.reverse_icl_weight
    if args.reverse_icl_final_weight < 0.0:
        parser.error('--reverse_icl_final_weight must be non-negative')
    if args.reverse_icl_final_weight > args.reverse_icl_weight:
        parser.error('--reverse_icl_final_weight cannot exceed --reverse_icl_weight')
    if args.reverse_icl_schedule == 'constant' and (
        args.reverse_icl_final_weight != args.reverse_icl_weight
    ):
        parser.error('constant reverse ICL schedule requires equal initial/final weights')
    if not 0 <= args.reverse_icl_step_epoch < args.epoch:
        parser.error('--reverse_icl_step_epoch must be in [0, epoch)')
    if not 0.0 <= args.icl_beta <= 1.0:
        parser.error('--icl_beta must be in [0, 1]')
    try:
        validate_rgat_configuration(args)
    except ValueError as error:
        parser.error(str(error))
    args.rgat_heads = MULTI_HEAD_DIM
    args.neighbor_size = (
        NEIGHBOR_SIZE + 1 if args.rgat_impl == PAPER_EXACT_RGAT else NEIGHBOR_SIZE
    )
    if args.run_name is None:
        setting = 'translated' if args.trans else 'original'
        args.run_name = '{}_{}_{}_{}_testbest_seed{}'.format(
            args.language, setting, args.profile, args.rgat_impl.replace('-', '_'), args.seed
        )
    return args


def validate_rgat_configuration(args):
    """Reject dimensions that the paper-described fusion layer cannot represent."""
    if args.gat_num < 1:
        raise ValueError('--gat_num must be at least 1')
    if MULTI_HEAD_DIM != 1:
        raise ValueError(
            'ICLEA fixes MULTI_HEAD_DIM=1 because the published 7x768 fusion dimension '
            'does not define a multi-head output width'
        )
    if args.rgat_impl == PAPER_EXACT_RGAT and args.gat_num != 1:
        raise ValueError('paper-exact-rgat requires the paper-configured one-hop GAT (--gat_num=1)')


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def validate_training_gpu_scope(cuda_visible_devices, visible_device_count):
    """Return the one allowed physical GPU or fail closed before training."""
    value = '' if cuda_visible_devices is None else cuda_visible_devices.strip()
    if value not in ('0', '1', '2', '3'):
        raise ValueError(
            'ICLEA training requires CUDA_VISIBLE_DEVICES to name exactly one physical GPU 0-3'
        )
    if int(visible_device_count) != 1:
        raise ValueError('ICLEA training requires exactly one visible CUDA device')
    return int(value)

def learning_rate_for_epoch(args, epoch):
    """Return the selected explicit learning-rate schedule."""
    if args.lr_schedule == 'constant':
        return args.lr
    if args.lr_schedule == 'author-periodic':
        return args.lr * (0.5 if (epoch + 1) % args.lr_step_size == 0 else 1.0)
    if args.lr_schedule == 'single-step':
        return args.lr * (args.lr_decay if epoch >= args.lr_step_size else 1.0)
    if args.lr_schedule == 'cosine':
        progress = min(max(float(epoch) / max(args.epoch - 1, 1), 0.0), 1.0)
        multiplier = args.lr_min_ratio + 0.5 * (1.0 - args.lr_min_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )
        return args.lr * multiplier
    completed_intervals = epoch // args.lr_step_size
    return args.lr * (args.lr_decay ** completed_intervals)


def adjust_learning_rate(optimizer, args, epoch):
    lr = learning_rate_for_epoch(args, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def temperature_for_epoch(args, epoch):
    """Return an explicit temperature schedule without changing the NCE/ICL form."""
    if args.temperature_schedule == 'constant':
        return args.t
    if args.temperature_schedule == 'step-increase':
        return args.temperature_final if epoch >= args.temperature_step_epoch else args.t
    progress = min(max(float(epoch) / max(args.epoch - 1, 1), 0.0), 1.0)
    return args.t + (args.temperature_final - args.t) * progress


def description_scale_for_epoch(args, epoch):
    """Return the explicit input-only description scale for one epoch."""
    if args.description_scale_schedule == 'constant':
        return args.description_scale
    if args.description_scale_schedule == 'step-increase':
        return (
            args.description_scale_final
            if epoch >= args.description_scale_step_epoch
            else args.description_scale
        )
    progress = max(
        0.0,
        float(epoch - args.description_scale_step_epoch)
        / max(args.epoch - 1 - args.description_scale_step_epoch, 1),
    )
    return args.description_scale + (
        args.description_scale_final - args.description_scale
    ) * progress


def reverse_icl_weight_for_epoch(args, epoch):
    """Schedule the optional reverse term without changing its contrastive loss."""
    if args.reverse_icl_schedule == 'constant':
        return args.reverse_icl_weight
    if args.reverse_icl_schedule == 'step-decay':
        return (
            args.reverse_icl_final_weight
            if epoch >= args.reverse_icl_step_epoch
            else args.reverse_icl_weight
        )
    progress = min(max(
        float(epoch - args.reverse_icl_step_epoch)
        / max(args.epoch - 1 - args.reverse_icl_step_epoch, 1),
        0.0,
    ), 1.0)
    return args.reverse_icl_weight + (
        args.reverse_icl_final_weight - args.reverse_icl_weight
    ) * progress


def metrics_improved(candidate, incumbent):
    """Select by test Hits@1, breaking an exact tie with test Hits@10."""
    if incumbent is None:
        return True
    return (candidate['hits1'], candidate['hits10']) > (
        incumbent['hits1'], incumbent['hits10']
    )


def joint_metrics_improved(candidate, incumbent):
    """Select by Hits@10 after a caller has enforced the Hits@1 floor."""
    if incumbent is None:
        return True
    return (candidate['hits10'], candidate['hits1']) > (
        incumbent['hits10'], incumbent['hits1']
    )


def mask_logits(target, mask):
    mask = mask.to(target.device)
    return target * mask + (1 - mask) * (-1e30)


def mine_top1(query, candidates, method='l2', csls_k=10, candidate_pool=100):
    """Return candidate row IDs and squared-L2 distances for pseudo-pair mining."""
    query = np.ascontiguousarray(query.astype(np.float32, copy=False))
    candidates = np.ascontiguousarray(candidates.astype(np.float32, copy=False))
    if method == 'l2':
        index = faiss.IndexFlatL2(candidates.shape[1])
        index.add(candidates)
        distances, neighbors = index.search(query, 1)
        return distances[:, 0], neighbors[:, 0]
    if method != 'csls':
        raise ValueError('Unknown pseudo-pair mining method: {}'.format(method))

    query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    candidate_norm = candidates / np.maximum(
        np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12
    )
    pool = min(int(candidate_pool), len(candidates))
    k_query = min(int(csls_k), len(candidates))
    k_candidate = min(int(csls_k), len(query))

    candidate_index = faiss.IndexFlatIP(candidate_norm.shape[1])
    candidate_index.add(candidate_norm)
    similarities, neighbors = candidate_index.search(query_norm, pool)
    query_density = similarities[:, :k_query].mean(axis=1)

    query_index = faiss.IndexFlatIP(query_norm.shape[1])
    query_index.add(query_norm)
    reverse_similarities, _ = query_index.search(candidate_norm, k_candidate)
    candidate_density = reverse_similarities.mean(axis=1)

    csls = 2.0 * similarities - query_density[:, None] - candidate_density[neighbors]
    best_column = csls.argmax(axis=1)
    best_neighbor = neighbors[np.arange(len(query)), best_column]
    delta = query - candidates[best_neighbor]
    squared_l2 = np.einsum('ij,ij->i', delta, delta)
    return squared_l2, best_neighbor


def mine_csls_bidirectional(vector_1, vector_2, csls_k=10, candidate_pool=100):
    """Mine both directions with two shared Faiss searches instead of four."""
    original_1 = np.ascontiguousarray(vector_1.astype(np.float32, copy=False))
    original_2 = np.ascontiguousarray(vector_2.astype(np.float32, copy=False))
    normalized_1 = original_1 / np.maximum(
        np.linalg.norm(original_1, axis=1, keepdims=True), 1e-12
    )
    normalized_2 = original_2 / np.maximum(
        np.linalg.norm(original_2, axis=1, keepdims=True), 1e-12
    )
    pool_12 = min(int(candidate_pool), len(normalized_2))
    pool_21 = min(int(candidate_pool), len(normalized_1))
    k_12 = min(int(csls_k), len(normalized_2))
    k_21 = min(int(csls_k), len(normalized_1))

    index_2 = faiss.IndexFlatIP(normalized_2.shape[1])
    index_2.add(np.ascontiguousarray(normalized_2))
    similarities_12, neighbors_12 = index_2.search(
        np.ascontiguousarray(normalized_1), pool_12
    )
    index_1 = faiss.IndexFlatIP(normalized_1.shape[1])
    index_1.add(np.ascontiguousarray(normalized_1))
    similarities_21, neighbors_21 = index_1.search(
        np.ascontiguousarray(normalized_2), pool_21
    )

    density_1 = similarities_12[:, :k_12].mean(axis=1)
    density_2 = similarities_21[:, :k_21].mean(axis=1)

    score_12 = 2.0 * similarities_12 - density_1[:, None] - density_2[neighbors_12]
    column_12 = score_12.argmax(axis=1)
    best_12 = neighbors_12[np.arange(len(original_1)), column_12]
    delta_12 = original_1 - original_2[best_12]
    distance_12 = np.einsum('ij,ij->i', delta_12, delta_12)

    score_21 = 2.0 * similarities_21 - density_2[:, None] - density_1[neighbors_21]
    column_21 = score_21.argmax(axis=1)
    best_21 = neighbors_21[np.arange(len(original_2)), column_21]
    delta_21 = original_2 - original_1[best_21]
    distance_21 = np.einsum('ij,ij->i', delta_21, delta_21)
    return (distance_12, best_12), (distance_21, best_21)

class NCESoftmaxLoss(nn.Module):

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()
        label = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        loss = self.criterion(x, label)
        return loss


class MyEmbedder(nn.Module):
    def __init__(self, args, vocab_size, padding=ord(' ')):
        super(MyEmbedder, self).__init__()

        self.args = args
        validate_rgat_configuration(self.args)

        self.attn = BatchMultiHeadGraphAttention(self.args)
        self.attn_relation = BatchMultiHeadGraphAttentionR(self.args)
        self.relation_tag = RelationAttention(self.args)
        path = join(DATA_DIR, 'DBP15K', self.args.language)
        relation_weight = torch.zeros(self.args.relation_padding_idx + 1, LaBSE_DIM)
        with open(join(path, 'jape_relation_emb_1.pkl'), 'rb') as f:    
            id_relation1 = pickle.load(f)
        with open(join(path, 'jape_relation_emb_2.pkl'), 'rb') as f:  
            id_relation2 = pickle.load(f)
        for relation_id, value in list(id_relation1.items()) + list(id_relation2.items()):
            vector = value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value
            relation_weight[int(relation_id)] = torch.as_tensor(vector).reshape(LaBSE_DIM)
        self.relation_emb = nn.Embedding.from_pretrained(
            relation_weight,
            freeze=False,
            padding_idx=self.args.relation_padding_idx,
        )
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(LaBSE_DIM * 7, LaBSE_DIM *5),
        )

        self.criterion = NCESoftmaxLoss()
        self.temperature = self.args.t
        self.description_scale = self.args.description_scale
        self.batch_queue = []
    


    def contrastive_loss(
        self,
        pos_1,
        pos_2,
        neg_value,
        positive_key_ids=None,
        negative_key_ids=None,
        queue_exclude_ids=None,
        in_batch_value=None,
        in_batch_key_ids=None,
        in_batch_exclude_ids=None,
    ):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        if queue_exclude_ids is None:
            queue_exclude_ids = positive_key_ids
        if queue_exclude_ids is not None and negative_key_ids is not None:
            queue_exclude_ids = queue_exclude_ids.to(l_neg.device).view(-1, 1)
            negative_key_ids = negative_key_ids.to(l_neg.device).view(1, -1)
            l_neg = l_neg.masked_fill(queue_exclude_ids == negative_key_ids, -1e30)
        if self.args.negative_set == 'paper-count' and bsz > 1:
            if in_batch_value is None:
                in_batch_value = pos_2
            in_batch_count = in_batch_value.shape[0]
            in_batch = torch.mm(
                pos_1.view(bsz, -1), in_batch_value.view(in_batch_count, -1).t()
            )
            if in_batch_exclude_ids is None:
                in_batch_exclude_ids = positive_key_ids
            if in_batch_key_ids is None:
                in_batch_key_ids = positive_key_ids
            if in_batch_exclude_ids is None or in_batch_key_ids is None:
                if in_batch_count != bsz:
                    raise ValueError(
                        'non-square in-batch negatives require explicit entity IDs'
                    )
                invalid = torch.eye(bsz, device=in_batch.device, dtype=torch.bool)
            else:
                exclude_ids = in_batch_exclude_ids.to(in_batch.device).view(-1, 1)
                key_ids = in_batch_key_ids.to(in_batch.device).view(1, -1)
                invalid = exclude_ids == key_ids
            in_batch = in_batch.masked_fill(invalid, -1e30)
            l_neg = torch.cat((in_batch, l_neg), dim=1)
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        temperature = getattr(self, 'temperature', self.args.t)
        return self.criterion(logits / temperature)
    
    def icl_loss(self, pos_1, pos_2, neg_value):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(pos_2.view(bsz, -1), neg_value.t())
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        temperature = getattr(self, 'temperature', self.args.t)
        return self.criterion(logits / temperature)

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def forward(self, batch):
        device = next(self.parameters()).device
        features, relation_ids, relation_id_mask, neighbor_mask = batch
        features = features.to(device, non_blocking=True)
        relation_ids = relation_ids.to(device, non_blocking=True)
        relation_id_mask = relation_id_mask.to(device, non_blocking=True)
        neighbor_mask = neighbor_mask.to(device, non_blocking=True)
        entity_width = LaBSE_DIM + DESC_DIM
        relation_start = entity_width + self.args.neighbor_size
        relation_adj_start = relation_start + LaBSE_DIM
        expected_width = relation_adj_start + self.args.neighbor_size
        if features.shape[1] != self.args.neighbor_size or features.shape[2] != expected_width:
            raise ValueError(
                'RGAT feature shape mismatch: expected (*, {}, {}), got {}'.format(
                    self.args.neighbor_size,
                    expected_width,
                    tuple(features.shape),
                )
            )
        center_in = features[:, :, :entity_width]
        description_scale = getattr(
            self, 'description_scale', self.args.description_scale
        )
        if description_scale != 1.0:
            center_in = torch.cat(
                (
                    center_in[:, :, :LaBSE_DIM],
                    center_in[:, :, LaBSE_DIM:] * description_scale,
                ),
                dim=2,
            )
        center_adj = features[:, :, entity_width:relation_start]
        relation_in = features[:, :, relation_start:relation_adj_start]
        relation_adj = features[:, :, relation_adj_start:expected_width]
        center = center_in[:, 0]
        center_neigh = center_in
        relation_v = self.relation_emb(relation_ids)
        relation_weights = relation_id_mask.unsqueeze(-1).to(relation_v.dtype)
        relation_v = (relation_v * relation_weights).sum(dim=2)
        relation_v = relation_v / relation_weights.sum(dim=2).clamp_min(1.0)

        for i in range(0, self.args.gat_num):
            center_neigh = self.attn(center_neigh, center_adj.bool()).squeeze(1)
            center_relation = self.attn_relation(relation_in, relation_adj.bool()).squeeze(1)
            center_relationtype = self.relation_tag(center_in, relation_v, neighbor_mask)

        center_neigh = center_neigh[:, 0]
        center_relation = center_relation[:,0]

        if self.args.center_norm:
            center = F.normalize(center, p=2, dim=1)
        if self.args.neighbor_norm:
            center_neigh = F.normalize(center_neigh, p=2, dim=1)
            center_relation = F.normalize(center_relation, p=2, dim=1)
            center_relationtype = F.normalize(center_relationtype, p=2, dim=1)
        if self.args.combine:
            out_hat = torch.cat((center, center_neigh,center_relation,center_relationtype), dim=1)
            out_hat = self.attn_mlp(out_hat)

            if self.args.emb_norm:
                out_hat = F.normalize(out_hat, p=2, dim=1)
        else:
            out_hat = center_neigh

        return out_hat


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, args, n_head=MULTI_HEAD_DIM, f_in=LaBSE_DIM+DESC_DIM, f_out=LaBSE_DIM+DESC_DIM, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.paper_exact = getattr(args, 'rgat_impl', AUTHOR_RGAT) == PAPER_EXACT_RGAT
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]  
        if self.n_head == 1:
            h_prime = torch.matmul(h, self.w[0]).unsqueeze(1)
        else:
            h_prime = torch.einsum('bni,hio->bhno', h, self.w)
        attention_input = h_prime if self.paper_exact else torch.tanh(h_prime)
        attn_src = torch.matmul(attention_input, self.a_src)
        attn_dst = torch.matmul(attention_input, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)
        attn = self.leaky_relu(attn)
        mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1], device=adj.device, dtype=torch.bool))
        attn = attn.masked_fill(mask, float("-inf"))
        attn = self.softmax(attn) 
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            output = output + self.bias
        if self.paper_exact:
            output = self.leaky_relu(output)
        return output

class BatchMultiHeadGraphAttentionR(nn.Module):
    def __init__(self, args, n_head=MULTI_HEAD_DIM, f_in=LaBSE_DIM, f_out=LaBSE_DIM, bias=True):
        super(BatchMultiHeadGraphAttentionR, self).__init__()
        self.n_head = n_head
        self.paper_exact = getattr(args, 'rgat_impl', AUTHOR_RGAT) == PAPER_EXACT_RGAT
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]  
        if self.n_head == 1:
            h_prime = torch.matmul(h, self.w[0]).unsqueeze(1)
        else:
            h_prime = torch.einsum('bni,hio->bhno', h, self.w)
        attention_input = h_prime if self.paper_exact else torch.tanh(h_prime)
        attn_src = torch.matmul(attention_input, self.a_src)
        attn_dst = torch.matmul(attention_input, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2) 
        attn = self.leaky_relu(attn)
        mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1], device=adj.device, dtype=torch.bool))
        attn = attn.masked_fill(mask, float("-inf"))
        attn = self.softmax(attn) 
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            output = output + self.bias
        if self.paper_exact:
            output = self.leaky_relu(output)
        return output

class RelationAttention(nn.Module):
    def __init__(self, args, n_head=1, in_dim = 768, hidden_dim = 768):
        super().__init__()
        self.paper_exact = getattr(args, 'rgat_impl', AUTHOR_RGAT) == PAPER_EXACT_RGAT
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.w = Parameter(torch.Tensor(1536, 1536))
        nn.init.xavier_uniform_(self.w)

    def forward(self, feature, relation_tag, neighbor_mask):
        Q = self.fc1(relation_tag)

        Q = self.relu(Q)
        Q = self.fc2(Q)
        if self.paper_exact:
            Q = self.relu(Q)
        Q = Q.squeeze(2)
        Q = Q.masked_fill(~neighbor_mask, -1e30)
        Q = F.softmax(Q, dim=1) * neighbor_mask.to(Q.dtype)
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Q = Q.unsqueeze(2)
        out = torch.bmm(feature.transpose(1, 2), Q)
        out = out.squeeze(2)
        out = torch.matmul(out, self.w)
        if self.paper_exact:
            out = self.relu(out)
        return out

class Trainer(object):
    def __init__(self, training=True):
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)
        self.seed = self.args.seed
        fix_seed(self.seed)
        print('Arguments:', vars(self.args))
        if training and not torch.cuda.is_available():
            raise RuntimeError('ICLEA training requires CUDA')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        self.physical_gpu = None
        if training:
            try:
                self.physical_gpu = validate_training_gpu_scope(
                    self.cuda_visible_devices, torch.cuda.device_count()
                )
            except ValueError as error:
                raise RuntimeError(str(error))
        loader1 = DBP15KRawDataset(
            self.args.language, "1", self.args.trans, neighbor_size=self.args.neighbor_size
        )
        self.args.relation_padding_idx = loader1.relation_padding_idx
        myset1 = MyRawdataset(
            loader1.id_neighbors_dict,
            loader1.id_neighbors_relation_dict,
            loader1.id_neighbors_desc_dict,
            loader1.id_adj_tensor_dict,
            loader1.id_relation_adj_tensor_dict,
            loader1.id_neighbors_relation_ids,
            loader1.id_neighbors_relation_mask,
            loader1.id_neighbor_mask,
        )
        del loader1
        self.dataset1 = myset1
        
        self.loader1 = Data.DataLoader(
            dataset=myset1, 
            batch_size=self.args.batch_size, 
            shuffle=False,
            drop_last=True,
        )

        self.eval_loader1 = Data.DataLoader(
            dataset=myset1,  
            batch_size=self.args.batch_size,  
            shuffle=False,
            drop_last=False,
        )

        del myset1

        loader2 = DBP15KRawDataset(
            self.args.language, "2", self.args.trans, neighbor_size=self.args.neighbor_size
        )
        if loader2.relation_padding_idx != self.args.relation_padding_idx:
            raise ValueError('The two KGs must share one relation padding index')
        myset2 = MyRawdataset(
            loader2.id_neighbors_dict,
            loader2.id_neighbors_relation_dict,
            loader2.id_neighbors_desc_dict,
            loader2.id_adj_tensor_dict,
            loader2.id_relation_adj_tensor_dict,
            loader2.id_neighbors_relation_ids,
            loader2.id_neighbors_relation_mask,
            loader2.id_neighbor_mask,
        )
        del loader2
        self.dataset2 = myset2

        self.loader2 = Data.DataLoader(
            dataset=myset2,  
            batch_size=self.args.batch_size,  
            shuffle=False,
            drop_last=True,
        )

        self.eval_loader2 = Data.DataLoader(
            dataset=myset2,  
            batch_size=self.args.batch_size,  
            shuffle=False,
            drop_last=False,
        )

        del myset2

        self.model = None
        self.iteration = 0
        def test_link_loader(mode):
            link_data = pd.read_csv(
                join(join(DATA_DIR, 'DBP15K', mode), 'test.ref'), sep='\t', header=None
            )
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            link_pair = []
            for i, _ in enumerate(entity1_id):
                link_pair.append([entity1_id[i], entity2_id[i]])
            return torch.tensor(link_pair, dtype=torch.long)

        # The requested main-experiment protocol explicitly monitors test.ref
        # throughout training and selects the epoch with maximum test Hits@1.
        self.test_link_tensor = test_link_loader(self.args.language)
        self.test_evaluations = 0
        self.neg_queue1 = []
        self.neg_queue2 = []

        self.id_list1 = []

        if training:
            self.model = MyEmbedder(self.args, VOCAB_SIZE)
            self.model = self.model.to(self.device)
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

            self._model = MyEmbedder(self.args, VOCAB_SIZE)
            if self.args.momentum_init == 'copy-online':
                self._model.load_state_dict(unwrap_model(self.model).state_dict())
            self._model = self._model.to(self.device)
            if torch.cuda.device_count() > 1:
                self._model = nn.DataParallel(self._model)
            for parameter in self._model.parameters():
                parameter.requires_grad = False
            self._model.eval()
            print('Model CUDA devices:', torch.cuda.device_count())
            self.iteration = 0
            self.lr = self.args.lr
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
            if self.args.lr_schedule == 'constant':
                self.scheduler_description = 'constant lr={}'.format(self.args.lr)
            elif self.args.lr_schedule == 'author-periodic':
                self.scheduler_description = (
                    'released-code periodic: half base LR every {}th epoch only'.format(
                        self.args.lr_step_size
                    )
                )
            elif self.args.lr_schedule == 'cosine':
                self.scheduler_description = (
                    'cosine gradual decay to min_ratio={} over {} epochs'.format(
                        self.args.lr_min_ratio, self.args.epoch
                    )
                )
            elif self.args.lr_schedule == 'single-step':
                self.scheduler_description = (
                    'single step at epoch {}: lr={} then {}'.format(
                        self.args.lr_step_size,
                        self.args.lr,
                        self.args.lr * self.args.lr_decay,
                    )
                )
            else:
                self.scheduler_description = (
                    'StepLR-equivalent: lr=base_lr*{}^(epoch//{}), updated at epoch start'.format(
                        self.args.lr_decay,
                        self.args.lr_step_size,
                    )
                )
            print('Learning-rate schedule:', self.scheduler_description)


    def save_model(self, model, epoch, batch_id, txt):
        os.makedirs(join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language), exist_ok=True)
        torch.save(model, join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language,
                               "model" + str(txt)
                               + "_epoch_" + str(epoch) 
                               + "_batchid" + str(batch_id)
                               + "_batch_size_" + str(BATCH_SIZE)
                               + "_neg_queue_len_" + str(self.args.queue_length - 1)))

    def _feature_rows(self, language_id, indices):
        dataset = self.dataset1 if language_id == 1 else self.dataset2
        rows = []
        for tensor in dataset.feature_tensors():
            tensor_indices = indices.reshape(-1).to(tensor.device, non_blocking=True)
            value = tensor.index_select(0, tensor_indices)
            if value.device != self.device:
                value = value.to(self.device, non_blocking=True)
            rows.append(value)
        return tuple(rows)

    @staticmethod
    def _batch_size(batch):
        return batch[0].shape[0]

    @staticmethod
    def _concat_batches(batches):
        return tuple(torch.cat(parts, dim=0) for parts in zip(*batches))

    def _encode_dataset(self, dataset):
        vectors = []
        total = len(dataset)
        for start in tqdm(range(0, total, self.args.eval_batch_size)):
            stop = min(start + self.args.eval_batch_size, total)
            token_data = tuple(tensor[start:stop] for tensor in dataset.feature_tensors())
            vectors.append(self.model(token_data).detach().cpu())
        return torch.cat(vectors, dim=0).numpy()

    @staticmethod
    def _link_indices(dataset1, dataset2, link_tensor):
        if (
            int(link_tensor[:, 0].max()) >= len(dataset1.id_to_index)
            or int(link_tensor[:, 1].max()) >= len(dataset2.id_to_index)
        ):
            raise ValueError('Alignment file contains an entity ID outside the encoded KG')
        source = dataset1.id_to_index[link_tensor[:, 0]]
        target = dataset2.id_to_index[link_tensor[:, 1]]
        if (source < 0).any() or (target < 0).any():
            raise ValueError('Alignment file contains an entity absent from the encoded KG')
        return source.numpy(), target.numpy()

    def _encode_both(self):
        was_training = self.model.training
        with torch.no_grad():
            self.model.eval()
            vector_1 = self._encode_dataset(self.dataset1)
            vector_2 = self._encode_dataset(self.dataset2)
        if was_training:
            self.model.train()
        return vector_1, vector_2

    def _cal_hits(self, vector_1, vector_2, link_tensor):
        source, target = self._link_indices(self.dataset1, self.dataset2, link_tensor)
        queries = np.ascontiguousarray(vector_1[source, :])
        candidates = np.ascontiguousarray(vector_2)
        index = faiss.IndexFlatL2(candidates.shape[1])
        index.add(candidates)
        _, neighbors = index.search(queries, 10)
        hit1 = float((neighbors[:, 0] == target).astype(np.int32).sum() / len(source))
        hit10 = float((neighbors == target[:, np.newaxis]).astype(np.int32).sum() / len(source))
        return {'num_queries': len(source), 'hits1': hit1, 'hits10': hit10}

    def evaluate_test(self, epoch):
        print('Test at epoch {} (complete target KG candidates)'.format(epoch))
        vector_1, vector_2 = self._encode_both()
        metrics = self._cal_hits(vector_1, vector_2, self.test_link_tensor)
        metrics['epoch'] = epoch
        self.test_evaluations += 1
        print('Test: epoch={} #Entity={} Hits@1={:.4f} Hits@10={:.4f}'.format(
            epoch, metrics['num_queries'], metrics['hits1'], metrics['hits10']))
        return metrics

    def get_pair(self):
        vector_1, vector_2 = self._encode_both()
        if self.args.pair_mining == 'csls':
            (distance_1, neighbor_1), (distance_2, neighbor_2) = mine_csls_bidirectional(
                vector_1,
                vector_2,
                csls_k=self.args.pair_csls_k,
                candidate_pool=self.args.pair_candidate_pool,
            )
        else:
            distance_1, neighbor_1 = mine_top1(vector_1, vector_2, method='l2')
            distance_2, neighbor_2 = mine_top1(vector_2, vector_1, method='l2')

        pair_1_to_2 = torch.from_numpy(neighbor_1.astype(np.int64))
        pair_2_to_1 = torch.from_numpy(neighbor_2.astype(np.int64))
        if self.args.pair_threshold is not None:
            pair_1_to_2[torch.from_numpy(distance_1 >= self.args.pair_threshold)] = -1
            pair_2_to_1[torch.from_numpy(distance_2 >= self.args.pair_threshold)] = -1

        def pair_statistics(mapping, distances):
            selected_mask = mapping >= 0
            selected = mapping[selected_mask]
            selected_distances = distances[selected_mask.numpy()]
            selected_count = int(selected.numel())
            unique_targets = int(torch.unique(selected).numel()) if selected_count else 0
            percentiles = (
                np.percentile(selected_distances, (50, 90, 95, 99)).tolist()
                if selected_count else [None, None, None, None]
            )
            return {
                'selected_pairs': selected_count,
                'unique_targets': unique_targets,
                'collision_queries': selected_count - unique_targets,
                'maximum_target_multiplicity': (
                    int(torch.bincount(selected).max()) if selected_count else 0
                ),
                'distance_percentiles': {
                    key: (float(value) if value is not None else None)
                    for key, value in zip(('p50', 'p90', 'p95', 'p99'), percentiles)
                },
            }

        pair_stats = {
            '1_to_2': pair_statistics(pair_1_to_2, distance_1),
            '2_to_1': pair_statistics(pair_2_to_1, distance_2),
        }

        print(
            'Pseudo pairs: {} + {} unique_targets={}+{} collisions={}+{} '
            '(mining={} threshold={})'.format(
                pair_stats['1_to_2']['selected_pairs'],
                pair_stats['2_to_1']['selected_pairs'],
                pair_stats['1_to_2']['unique_targets'],
                pair_stats['2_to_1']['unique_targets'],
                pair_stats['1_to_2']['collision_queries'],
                pair_stats['2_to_1']['collision_queries'],
                self.args.pair_mining,
                self.args.pair_threshold,
            )
        )
        return pair_1_to_2, pair_2_to_1, pair_stats

    def _checkpoint_path(self, filename='best_test_hits1.pt'):
        path = join(self.args.output_dir, 'checkpoints', self.args.run_name)
        os.makedirs(path, exist_ok=True)
        return join(path, filename)

    def _checkpoint_payload(self, epoch, test_metrics):
        return {
            'online_model': unwrap_model(self.model).state_dict(),
            'momentum_model': unwrap_model(self._model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'test': test_metrics,
            'arguments': vars(self.args),
        }

    def _write_result(self, result):
        result_dir = join(self.args.output_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)
        result_path = join(result_dir, self.args.run_name + '.json')
        temporary_path = result_path + '.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(temporary_path, result_path)
        print('Result JSON:', result_path)
        return result_path

    def _base_result(self, status, total_steps, icl_steps, started_at):
        return {
            'status': status,
            'run_name': self.args.run_name,
            'language': self.args.language,
            'setting': 'translated' if self.args.trans else 'original',
            'profile': self.args.profile,
            'rgat_impl': self.args.rgat_impl,
            'seed': self.args.seed,
            'protocol': {
                'pair_threshold': self.args.pair_threshold,
                'pair_mining': self.args.pair_mining,
                'pair_csls_k': self.args.pair_csls_k,
                'pair_candidate_pool': self.args.pair_candidate_pool,
                'warmup': self.args.warmup,
                'selection_protocol': 'test_best',
                'validation_access': 'none',
                'test_candidates': 'complete_target_kg',
                'evaluation_distance': 'faiss_squared_l2',
                'test_access': 'monitored_during_training',
                'selection_metric': 'test_hits1_then_hits10',
                'reported_metric': 'best_test_over_training',
                'test_evaluations': self.test_evaluations,
                'joint_hits1_floor': self.args.joint_hits1_floor,
                'snapshot_epochs': self.args.snapshot_epochs,
                'lr_schedule': self.scheduler_description,
                'lr_schedule_name': self.args.lr_schedule,
                'lr_initial': self.args.lr,
                'lr_min_ratio': self.args.lr_min_ratio,
                'lr_step_size': self.args.lr_step_size,
                'lr_decay': self.args.lr_decay,
                'temperature_schedule': self.args.temperature_schedule,
                'temperature_initial': self.args.t,
                'temperature_final': self.args.temperature_final,
                'temperature_step_epoch': self.args.temperature_step_epoch,
                'rgat_impl': self.args.rgat_impl,
                'rgat_heads': self.args.rgat_heads,
                'gat_layers': self.args.gat_num,
                'input_slots': self.args.neighbor_size,
                'maximum_neighbors': self.args.neighbor_size - 1,
                'batch_order': self.args.batch_order,
                'negative_set': self.args.negative_set,
                'description_scale': self.args.description_scale,
                'description_scale_schedule': self.args.description_scale_schedule,
                'description_scale_final': self.args.description_scale_final,
                'description_scale_step_epoch': self.args.description_scale_step_epoch,
                'icl_beta': self.args.icl_beta,
                'icl_source_inbatch': self.args.icl_source_inbatch,
                'reverse_icl_weight': self.args.reverse_icl_weight,
                'reverse_icl_schedule': self.args.reverse_icl_schedule,
                'reverse_icl_final_weight': self.args.reverse_icl_final_weight,
                'reverse_icl_step_epoch': self.args.reverse_icl_step_epoch,
                'online_dropout': self.args.dropout,
                'exclude_false_negatives': self.args.exclude_false_negatives,
                'momentum_init': self.args.momentum_init,
                'cuda_visible_devices': self.cuda_visible_devices,
                'visible_cuda_device_count': torch.cuda.device_count(),
                'physical_gpu': self.physical_gpu,
            },
            'optimizer_steps': total_steps,
            'icl_steps': icl_steps,
            'wall_seconds': time.time() - started_at,
            'arguments': vars(self.args),
        }

    def train(self, start=0):
        """Train all epochs and report the epoch with maximum monitored test Hits@1."""
        fix_seed(self.seed)
        started_at = time.time()
        all_data_batches = []
        if self.args.feature_device == 'gpu':
            self.dataset1.to(self.device)
            self.dataset2.to(self.device)
        print('Feature tensors:', self.dataset1.x_train.device, self.dataset2.x_train.device)

        for index_data, _ in self.loader1:
            all_data_batches.append((1, index_data))
        for index_data, _ in self.loader2:
            all_data_batches.append((2, index_data))
        if self.args.batch_order == 'fixed':
            random.shuffle(all_data_batches)

        best_test = None
        best_joint_test = None
        test_history = []
        best_epoch = None
        best_joint_epoch = None
        checkpoint_path = self._checkpoint_path()
        joint_checkpoint_path = self._checkpoint_path('best_joint_hits10.pt')
        pair_1_to_2 = None
        pair_2_to_1 = None
        pseudo_pair_history = []
        total_steps = 0
        icl_steps = 0
        optimizer_start = None
        torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(start, self.args.epoch):
            current_lr = adjust_learning_rate(self.optimizer, self.args, epoch)
            current_temperature = temperature_for_epoch(self.args, epoch)
            current_description_scale = description_scale_for_epoch(self.args, epoch)
            current_reverse_icl_weight = reverse_icl_weight_for_epoch(self.args, epoch)
            unwrap_model(self.model).temperature = current_temperature
            unwrap_model(self._model).temperature = current_temperature
            unwrap_model(self.model).description_scale = current_description_scale
            unwrap_model(self._model).description_scale = current_description_scale
            print(
                'Epoch {} learning_rate={:.12g} temperature={:.12g} '
                'description_scale={:.12g} reverse_icl_weight={:.12g}'.format(
                    epoch,
                    current_lr,
                    current_temperature,
                    current_description_scale,
                    current_reverse_icl_weight,
                )
            )
            if epoch >= self.args.warmup:
                pair_1_to_2, pair_2_to_1, pair_stats = self.get_pair()
                pair_stats['epoch'] = epoch
                pseudo_pair_history.append(pair_stats)

            if self.args.batch_order == 'reshuffle':
                random.shuffle(all_data_batches)
            epoch_start = time.time()
            for batch_id, (language_id, index_data) in tqdm(
                enumerate(all_data_batches), total=len(all_data_batches)
            ):
                queue = self.neg_queue1 if language_id == 1 else self.neg_queue2
                queue.append(index_data)
                if len(queue) < self.args.queue_length + 1:
                    continue
                if len(queue) > self.args.queue_length + 1:
                    raise RuntimeError('Negative queue exceeded its configured length')

                pos_batch_index = queue.pop(0)
                neg_queue_index = torch.cat(queue, dim=0)
                pos_batch = self._feature_rows(language_id, pos_batch_index)
                neg_batch = self._feature_rows(language_id, neg_queue_index)

                pair_map = pair_1_to_2 if language_id == 1 else pair_2_to_1
                target_language_id = 2 if language_id == 1 else 1
                target_queue = self.neg_queue2 if language_id == 1 else self.neg_queue1
                valid_pair_mask = None
                pair2_batch = None
                neg_target_batch = None
                if pair_map is not None and len(target_queue) == self.args.queue_length:
                    pair2_index = pair_map[pos_batch_index]
                    valid_pair_mask = pair2_index >= 0
                    if int(valid_pair_mask.sum()) > 1:
                        pair2_batch = self._feature_rows(
                            target_language_id, pair2_index[valid_pair_mask]
                        )
                        neg_target_batch = self._feature_rows(
                            target_language_id, torch.cat(target_queue, dim=0)
                        )
                    else:
                        valid_pair_mask = None

                self.optimizer.zero_grad(set_to_none=True)
                if optimizer_start is None:
                    optimizer_start = time.time()
                pos_1 = self.model(pos_batch)
                momentum_batches = [pos_batch, neg_batch]
                if pair2_batch is not None:
                    momentum_batches.extend([pair2_batch, neg_target_batch])
                with torch.no_grad():
                    self._model.eval()
                    if self.args.fused_momentum:
                        momentum_sizes = [self._batch_size(batch) for batch in momentum_batches]
                        momentum_output = self._model(self._concat_batches(momentum_batches))
                        momentum_output = momentum_output.split(momentum_sizes, dim=0)
                    else:
                        momentum_output = [self._model(batch) for batch in momentum_batches]

                pos_2, neg_value = momentum_output[:2]
                online_model = unwrap_model(self.model)
                use_id_mask = self.args.exclude_false_negatives
                loss = online_model.contrastive_loss(
                    pos_1,
                    pos_2,
                    neg_value,
                    positive_key_ids=pos_batch_index if use_id_mask else None,
                    negative_key_ids=neg_queue_index if use_id_mask else None,
                )
                if pair2_batch is not None:
                    icl_steps += 1
                    pair2_momentum, neg_value_target = momentum_output[2:]
                    pair1_online = pos_1[valid_pair_mask.to(pos_1.device)]
                    pair2_index_valid = pair2_index[valid_pair_mask]
                    neg_target_index = torch.cat(target_queue, dim=0)
                    source_negative_loss = online_model.contrastive_loss(
                        pair1_online,
                        pair2_momentum,
                        neg_value,
                        positive_key_ids=pair2_index_valid if use_id_mask else None,
                        negative_key_ids=neg_queue_index if use_id_mask else None,
                        queue_exclude_ids=pos_batch_index[valid_pair_mask]
                        if use_id_mask else None,
                        in_batch_value=(
                            pos_2
                            if self.args.icl_source_inbatch == 'source'
                            else None
                        ),
                        in_batch_key_ids=(
                            pos_batch_index
                            if self.args.icl_source_inbatch == 'source'
                            else None
                        ),
                        in_batch_exclude_ids=(
                            pos_batch_index[valid_pair_mask]
                            if self.args.icl_source_inbatch == 'source'
                            else None
                        ),
                    )
                    target_negative_loss = online_model.contrastive_loss(
                        pair1_online,
                        pair2_momentum,
                        neg_value_target,
                        positive_key_ids=pair2_index_valid if use_id_mask else None,
                        negative_key_ids=neg_target_index if use_id_mask else None,
                    )
                    loss = (
                        loss
                        + self.args.icl_beta * source_negative_loss
                        + (1.0 - self.args.icl_beta) * target_negative_loss
                    )
                    if current_reverse_icl_weight > 0.0:
                        pair2_online = self.model(pair2_batch)
                        pair1_momentum = pos_2[valid_pair_mask.to(pos_2.device)]
                        reverse_pair_loss = online_model.contrastive_loss(
                            pair2_online,
                            pair1_momentum,
                            neg_value,
                            positive_key_ids=pos_batch_index[valid_pair_mask]
                            if use_id_mask else None,
                            negative_key_ids=neg_queue_index if use_id_mask else None,
                        )
                        loss = loss + current_reverse_icl_weight * reverse_pair_loss

                loss.backward()
                self.optimizer.step()
                unwrap_model(self._model).update(unwrap_model(self.model))
                self.iteration += 1
                total_steps += 1
                if total_steps % 200 == 0:
                    print('epoch={} batch={} step={} loss={:.6f}'.format(
                        epoch, batch_id, total_steps, float(loss.detach().cpu())))

                if self.args.max_steps and total_steps >= self.args.max_steps:
                    elapsed = time.time() - optimizer_start
                    result = self._base_result('smoke', total_steps, icl_steps, started_at)
                    result['throughput_steps_per_second'] = total_steps / elapsed
                    result['peak_cuda_mib'] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
                    self._write_result(result)
                    print(
                        'Reached max_steps={} icl_steps={} elapsed={:.3f}s '
                        'steps_per_second={:.3f} peak_cuda_mib={:.1f}; test was not evaluated'.format(
                            self.args.max_steps,
                            icl_steps,
                            elapsed,
                            total_steps / elapsed,
                            result['peak_cuda_mib'],
                        )
                    )
                    return result

            print('Epoch {} completed in {:.1f}s (optimizer steps={})'.format(
                epoch, time.time() - epoch_start, total_steps))
            should_evaluate = (
                (epoch + 1) % self.args.eval_every_epochs == 0
                or epoch + 1 == self.args.epoch
            )
            if not should_evaluate:
                continue

            test_metrics = self.evaluate_test(epoch)
            test_metrics['learning_rate'] = current_lr
            test_metrics['temperature'] = current_temperature
            test_metrics['description_scale'] = current_description_scale
            test_metrics['reverse_icl_weight'] = current_reverse_icl_weight
            test_history.append(test_metrics)
            checkpoint_payload = None
            if metrics_improved(test_metrics, best_test):
                best_test = test_metrics
                best_epoch = epoch
                checkpoint_payload = self._checkpoint_payload(epoch, test_metrics)
                torch.save(checkpoint_payload, checkpoint_path)
                print(
                    'New best test checkpoint: epoch={} Hits@1={:.4f} Hits@10={:.4f} {}'.format(
                        epoch, best_test['hits1'], best_test['hits10'], checkpoint_path
                    )
                )
            if (
                self.args.joint_hits1_floor is not None
                and test_metrics['hits1'] >= self.args.joint_hits1_floor
                and joint_metrics_improved(test_metrics, best_joint_test)
            ):
                best_joint_test = test_metrics
                best_joint_epoch = epoch
                if checkpoint_payload is None:
                    checkpoint_payload = self._checkpoint_payload(epoch, test_metrics)
                torch.save(checkpoint_payload, joint_checkpoint_path)
                print(
                    'New constrained Hits@10 checkpoint: epoch={} Hits@1={:.4f} '
                    'Hits@10={:.4f} {}'.format(
                        epoch,
                        best_joint_test['hits1'],
                        best_joint_test['hits10'],
                        joint_checkpoint_path,
                    )
                )
            if epoch in self.args.snapshot_epochs:
                if checkpoint_payload is None:
                    checkpoint_payload = self._checkpoint_payload(epoch, test_metrics)
                snapshot_path = self._checkpoint_path('snapshot_epoch_{:03d}.pt'.format(epoch))
                torch.save(checkpoint_payload, snapshot_path)
                print('Archived diagnostic snapshot: {}'.format(snapshot_path))

        if best_test is None:
            raise RuntimeError('Training completed without a test evaluation')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        unwrap_model(self.model).load_state_dict(checkpoint['online_model'])
        result = self._base_result('complete', total_steps, icl_steps, started_at)
        result.update({
            'best_epoch': best_epoch,
            'best_test': best_test,
            'test': best_test,
            'test_history': test_history,
            'pseudo_pair_history': pseudo_pair_history,
            'trained_epochs': self.args.epoch - start,
            'stopped_early': False,
            'checkpoint': checkpoint_path,
            'joint_hits1_floor': self.args.joint_hits1_floor,
            'best_joint_epoch': best_joint_epoch,
            'best_joint_test': best_joint_test,
            'joint_checkpoint': (
                joint_checkpoint_path if best_joint_test is not None else None
            ),
            'peak_cuda_mib': torch.cuda.max_memory_allocated(self.device) / 1024 ** 2,
        })
        self._write_result(result)
        print('FINAL_RESULT_JSON=' + json.dumps(result, sort_keys=True))
        return result
