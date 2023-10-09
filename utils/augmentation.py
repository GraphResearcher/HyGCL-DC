# -*- coding:utf-8 -*-

import numpy as np
import copy
import numpy.random as random

def mask_nodes(dataset, ratio):
    node_num = dataset.X.shape[0]
    mask_node_num = int(node_num * ratio)
    mask_node_index = np.random.choice(node_num, mask_node_num, replace=False)
    mask = random.rand(mask_node_num, dataset.X.shape[1])
    dataset.X[mask_node_index] = dataset.X[mask_node_index] + mask
    dataset.update()
    return dataset


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

