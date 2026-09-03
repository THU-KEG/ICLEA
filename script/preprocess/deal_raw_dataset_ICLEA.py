import torch
from torch.utils.data import Dataset


class MyRawdataset(Dataset):
    """Dense feature store indexed by compact row IDs."""

    def __init__(
        self,
        id_features_dict,
        id_relation_dict,
        id_desc_dict,
        adj_tensor_dict,
        id_relation_adj_tensor_dict,
        id_neighbors_relation_ids,
        id_neighbors_relation_mask,
        id_neighbor_mask,
    ):
        super(MyRawdataset, self).__init__()
        self.entity_ids = list(id_features_dict.keys())
        required_dicts = (
            id_relation_dict,
            id_desc_dict,
            adj_tensor_dict,
            id_relation_adj_tensor_dict,
            id_neighbors_relation_ids,
            id_neighbors_relation_mask,
            id_neighbor_mask,
        )
        if any(set(values) != set(self.entity_ids) for values in required_dicts):
            raise ValueError('Feature dictionaries do not contain the same entity IDs')

        entity = torch.stack([torch.as_tensor(id_features_dict[k]) for k in self.entity_ids]).float()
        desc = torch.stack([torch.as_tensor(id_desc_dict[k]) for k in self.entity_ids]).float()
        entity_adj = torch.stack([torch.as_tensor(adj_tensor_dict[k]) for k in self.entity_ids]).float()
        relation = torch.stack([torch.as_tensor(id_relation_dict[k]) for k in self.entity_ids]).float()
        relation_adj = torch.stack(
            [torch.as_tensor(id_relation_adj_tensor_dict[k]) for k in self.entity_ids]
        ).float()
        self.x_train = torch.cat((entity, desc, entity_adj, relation, relation_adj), dim=2)
        self.relation_ids = torch.stack(
            [torch.as_tensor(id_neighbors_relation_ids[k]) for k in self.entity_ids]
        ).long()
        self.relation_id_mask = torch.stack(
            [torch.as_tensor(id_neighbors_relation_mask[k]) for k in self.entity_ids]
        ).bool()
        self.neighbor_mask = torch.stack(
            [torch.as_tensor(id_neighbor_mask[k]) for k in self.entity_ids]
        ).bool()

        self.num = len(self.entity_ids)
        self.y_train = torch.tensor(self.entity_ids, dtype=torch.long).view(-1, 1)
        self.x_index = torch.arange(self.num, dtype=torch.long)
        self.index_to_id = self.y_train.view(-1).clone()
        self.id_to_index = torch.full(
            (int(self.index_to_id.max().item()) + 1,), -1, dtype=torch.long
        )
        self.id_to_index[self.index_to_id] = self.x_index
        print(
            'features={} relation_ids={} masks={}'.format(
                tuple(self.x_train.shape),
                tuple(self.relation_ids.shape),
                tuple(self.neighbor_mask.shape),
            )
        )

    def feature_tensors(self):
        return (
            self.x_train,
            self.relation_ids,
            self.relation_id_mask,
            self.neighbor_mask,
        )

    def to(self, device):
        self.x_train = self.x_train.to(device)
        self.relation_ids = self.relation_ids.to(device)
        self.relation_id_mask = self.relation_id_mask.to(device)
        self.neighbor_mask = self.neighbor_mask.to(device)
        return self

    def __getitem__(self, index):
        return self.x_index[index], self.y_train[index]

    def __len__(self):
        return self.num
