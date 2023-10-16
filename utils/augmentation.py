# -*- coding:utf-8 -*-

import numpy as np
import copy
import numpy.random as random
import torch
from collections import defaultdict

# def mask_nodes(dataset, ratio):
#     node_num = dataset.X.shape[0]
#     mask_node_num = int(node_num * ratio)
#     mask_node_index = np.random.choice(node_num, mask_node_num, replace=False)
#     mask = random.rand(mask_node_num, dataset.X.shape[1])
#     dataset.X[mask_node_index] = dataset.X[mask_node_index] + mask
#     dataset.update()
#     return dataset


def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.X.shape
    mask_num = int(node_num * aug_ratio)

    token = np.mean(data.X,axis=0).reshape(1,-1)
    # zero_v = torch.zeros_like(token)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.X[idx_mask] = token
    data.update()
    return data


# def permute_hyperedges(data, aug_ratio):
#     node_num, _ = data.X.size()
#     # node_num, _ = data.X.shape
#     _, edge_num = data.edge_index.size()
#     # edge_num,_ = data.H.shape
#     hyperedge_num = int(data.num_hyperedges[0].item())
#     permute_num = int(hyperedge_num * aug_ratio)
#     index = defaultdict(list)
#     edge_index = data.edge_index.cpu().numpy()
#     edge_remove_index = np.random.choice(hyperedge_num, permute_num, replace=False)
#     edge_remove_index_dict = {ind: i for i, ind in enumerate(edge_remove_index)}
#
#     edge_remove_index_all = [i for i, he in enumerate(edge_index[1]) if he in edge_remove_index_dict]
#     # print(len(edge_remove_index_all), edge_num, len(edge_remove_index), aug_ratio, hyperedge_num)
#     edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
#     edge_after_remove = edge_index[:, edge_keep_index]
#     edge_index = edge_after_remove
#
#     data.edge_index = torch.tensor(edge_index)
#     return data

def permute_hyperedges(dataset, aug_ratio):
    hyperedge_num = dataset.H.shape[0]
    permute_num = int(hyperedge_num * aug_ratio)
    keep_hyperedge_num = hyperedge_num - permute_num
    edge_keep_index = np.random.choice(hyperedge_num, keep_hyperedge_num, replace=False)
    dataset.H = dataset.H[edge_keep_index]
    dataset.update()
    return dataset


def aug(dataset, method, ratio, deepcopy=False):
    if type(method) is str:
        data_aug = copy.deepcopy(dataset) if deepcopy else dataset
        if method == "mask":
            data_aug = mask_nodes(data_aug, ratio)
        elif method == "hyperedge":
            data_aug = permute_hyperedges(data_aug, ratio)
        else:
            raise ValueError(f'not supported augmentation')
    elif type(method) is list:
        data_aug = copy.deepcopy(dataset) if deepcopy else dataset
        for m in method:
            data_aug = aug(dataset, m, False)
    else:
        raise ValueError("Unknown input methods")
    return data_aug

