from collections import OrderedDict
from os.path import join

import pandas as pd
import pickle
import torch

from settings import DATA_DIR, DESC_DIM, LaBSE_DIM, NEIGHBOR_SIZE


def _vector(value, dim, label):
    """Return an archived pickle vector and validate its dimensionality."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    tensor = torch.as_tensor(value)
    if tensor.numel() != dim:
        raise ValueError('{} must contain {} values, got {}'.format(label, dim, tensor.numel()))
    return tensor.float().reshape(dim)


class DBP15KRawDataset(object):
    """Materialize aligned center/neighbor/relation slots for one DBP15K KG.

    Slot 0 is the center entity. Every later valid slot is one neighbor and
    carries exactly the relation IDs connecting that neighbor to the center.
    The archived loader separately deduplicated neighbors and relations, which
    destroyed this one-to-one slot correspondence.
    """

    def __init__(self, language, doc_id, trans, neighbor_size=NEIGHBOR_SIZE):
        self.language = language
        self.trans = trans
        self.doc_id = doc_id
        if neighbor_size < 2:
            raise ValueError('neighbor_size must include a center and at least one neighbor slot')
        self.neighbor_size = int(neighbor_size)
        self.path = join(DATA_DIR, 'DBP15K', self.language)
        self.id_entity = {}
        self.id_relation = {}
        self.jape_map = {}
        self.id_entity_desc = {}
        self.id_adj_tensor_dict = {}
        self.id_neighbors_dict = {}
        self.id_neighbors_relation_dict = {}
        self.id_relation_adj_tensor_dict = {}
        self.id_neighbors_relation_ids = {}
        self.id_neighbors_relation_mask = {}
        self.id_neighbor_mask = {}
        self.id_neighbors_desc_dict = {}
        self.id_desc_adj_tensor_dict = {}
        self.load()
        self.load_relation()
        self.mapping_jape_to_hy()
        self.relation_padding_idx = self._relation_padding_idx()
        self.max_relations_per_neighbor = self._max_relations_per_neighbor()
        self.id_neighbors_loader()

    def load(self):
        if not self.trans:
            with open(join(self.path, 'raw_LaBSE_emb_' + self.doc_id + '.pkl'), 'rb') as f:
                self.id_entity = pickle.load(f)
            with open(join(self.path, 'desc_LaBSE_emb_' + self.doc_id + '.pkl'), 'rb') as f:
                self.id_entity_desc = pickle.load(f)
        else:
            with open(join(self.path, 'transLaBSE_name_emb_' + self.doc_id + '.pkl'), 'rb') as f:
                self.id_entity = pickle.load(f)
            with open(join(self.path, 'trans_desc_LaBSE_emb_' + self.doc_id + '.pkl'), 'rb') as f:
                self.id_entity_desc = pickle.load(f)

    def load_relation(self):
        with open(join(self.path, 'jape_relation_emb_' + self.doc_id + '.pkl'), 'rb') as f:
            self.id_relation = pickle.load(f)

    def mapping_jape_to_hy(self):
        with open(join(self.path, 'jape_hy_' + self.doc_id), 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.rstrip('\n').split('\t')
                self.jape_map[int(fields[0])] = int(fields[1])

    def _relation_padding_idx(self):
        """Use one shared padding ID across both KGs of a language pair."""
        maximum = -1
        for side in ('1', '2'):
            with open(join(self.path, 'jape_relation_emb_' + side + '.pkl'), 'rb') as f:
                relation_dict = pickle.load(f)
            if relation_dict:
                maximum = max(maximum, max(int(key) for key in relation_dict))
        return maximum + 1

    def _max_relations_per_neighbor(self):
        """Find a common exact width for parallel edges in the two KGs."""
        maximum = 1
        for side in ('1', '2'):
            triples = pd.read_csv(
                join(self.path, 'jape_triples_' + side),
                header=None,
                sep='\t',
                names=['head', 'relation', 'tail'],
            )
            pair_relations = {}
            for row in triples.itertuples(index=False):
                pair = tuple(sorted((int(row.head), int(row.tail))))
                pair_relations.setdefault(pair, set()).add(int(row.relation))
            if pair_relations:
                maximum = max(maximum, max(len(values) for values in pair_relations.values()))
        return maximum

    def get_adj(self, valid_len):
        adj = torch.zeros(self.neighbor_size, self.neighbor_size, dtype=torch.bool)
        adj[0, :valid_len] = True
        adj[:valid_len, 0] = True
        diagonal = torch.arange(valid_len)
        adj[diagonal, diagonal] = True
        return adj

    def id_neighbors_loader(self):
        triples = pd.read_csv(
            join(self.path, 'jape_triples_' + self.doc_id),
            header=None,
            sep='\t',
            names=['head', 'relation', 'tail'],
        )

        edges = {int(entity_id): OrderedDict() for entity_id in self.id_entity}
        for row in triples.itertuples(index=False):
            head = int(self.jape_map[int(row.head)])
            tail = int(self.jape_map[int(row.tail)])
            relation = int(row.relation)
            if head not in edges or tail not in edges:
                raise ValueError('Triple references an entity absent from the feature pickles')
            edges[head].setdefault(tail, [])
            edges[tail].setdefault(head, [])
            if relation not in edges[head][tail]:
                edges[head][tail].append(relation)
            if relation not in edges[tail][head]:
                edges[tail][head].append(relation)

        zero_entity = torch.zeros(LaBSE_DIM)
        zero_desc = torch.zeros(DESC_DIM)
        zero_relation = torch.zeros(LaBSE_DIM)

        for center_key in self.id_entity:
            center_id = int(center_key)
            neighbors = list(edges[center_id].items())[: self.neighbor_size - 1]
            valid_len = 1 + len(neighbors)
            slot_entity = [_vector(self.id_entity[center_key], LaBSE_DIM, 'entity name')]
            slot_desc = [_vector(self.id_entity_desc[center_key], DESC_DIM, 'entity description')]
            slot_relation = [_vector(self.id_entity[center_key], LaBSE_DIM, 'center name')]
            relation_ids = torch.full(
                (self.neighbor_size, self.max_relations_per_neighbor),
                self.relation_padding_idx,
                dtype=torch.long,
            )
            relation_mask = torch.zeros_like(relation_ids, dtype=torch.bool)
            neighbor_mask = torch.zeros(self.neighbor_size, dtype=torch.bool)

            for slot, (neighbor_id, rel_ids) in enumerate(neighbors, start=1):
                slot_entity.append(_vector(self.id_entity[neighbor_id], LaBSE_DIM, 'neighbor name'))
                slot_desc.append(_vector(self.id_entity_desc[neighbor_id], DESC_DIM, 'neighbor description'))
                relation_vectors = [
                    _vector(self.id_relation[relation_id], LaBSE_DIM, 'relation name')
                    for relation_id in rel_ids
                ]
                slot_relation.append(torch.stack(relation_vectors).mean(dim=0))
                relation_ids[slot, : len(rel_ids)] = torch.tensor(rel_ids, dtype=torch.long)
                relation_mask[slot, : len(rel_ids)] = True
                neighbor_mask[slot] = True

            padding = self.neighbor_size - valid_len
            slot_entity.extend([zero_entity] * padding)
            slot_desc.extend([zero_desc] * padding)
            slot_relation.extend([zero_relation] * padding)
            adj = self.get_adj(valid_len)

            self.id_neighbors_dict[center_id] = torch.stack(slot_entity)
            self.id_neighbors_desc_dict[center_id] = torch.stack(slot_desc)
            self.id_neighbors_relation_dict[center_id] = torch.stack(slot_relation)
            self.id_adj_tensor_dict[center_id] = adj
            self.id_desc_adj_tensor_dict[center_id] = adj.clone()
            self.id_relation_adj_tensor_dict[center_id] = adj.clone()
            self.id_neighbors_relation_ids[center_id] = relation_ids
            self.id_neighbors_relation_mask[center_id] = relation_mask
            self.id_neighbor_mask[center_id] = neighbor_mask

        print(
            'Loaded KG{}: entities={} max_parallel_relations={} relation_padding_idx={}'.format(
                self.doc_id,
                len(self.id_neighbors_dict),
                self.max_relations_per_neighbor,
                self.relation_padding_idx,
            )
        )
