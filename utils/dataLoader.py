# -*- coding:utf-8 -*-
import torch
import numpy as np
from itertools import combinations
import numpy.random as random
import os.path as osp


class hypergraph_data():
    def __init__(self, X, Y, H, symmetric=True):
        self.X, self.Y, self.H = X, Y, H
        self.XE, self.E, self.M, self.L = None, None, None, None
        self.X_TORCH, self.Y_TORCH, self.E_TORCH, self.M_TORCH, self.XE_TORCH = None, None, None, None, None
        self.device = None

        self.symmetric = symmetric

        self.update_edges()
        self.train_idx, self.test_idx = [], []

        self.init = True
        self.name = None
        self.masked_nodes = []

    def update(self):
        self.num_hyperedges = self.H.shape[0]
        self.n = self.Y.shape[0]
        self.n_cls = self.Y.shape[1]
        self.n_features = self.X.shape[1]
        # self.XE = normalise(self.XE)
        self.X_TORCH = torch.FloatTensor(self.X).to(self.device)
        self.Y_TORCH = torch.Tensor(self.Y).float().to(self.device)
        self.XE_TORCH = torch.Tensor(self.XE).float().to(self.device)
        self.M_TORCH = torch.Tensor(self.M).float().to(self.device)
        self.E_TORCH = self.M_TORCH.nonzero().T.to(self.device).type(torch.int64)
        self.E = self.E_TORCH.cpu().numpy()
        self.L = [h.nonzero()[0].tolist() for h in self.H]
        self.n_edges = self.E.shape[0]

    def update_edges(self):
        self.clique(self.H)
        self.update()

    def clique(self, H):
        M = np.zeros((H.shape[1], H.shape[1]))
        for e in H:
            nodes = np.where(e == 1)[0].tolist()
            for v, u in combinations(nodes, 2):
                if u <= v:
                    M[u, v] = 1
                else:
                    M[v, u] = 1
        if self.symmetric:
            M = np.maximum(M, M.T)
        self.M = M
        self.E = np.concatenate(([M.nonzero()[0]], [M.nonzero()[1]]), axis=0).T
        self.XE = self.X

    def split_data(self, ratio):
        self.train_idx, self.test_idx = [], []
        train_idx = []
        for i in range(self.n_cls):
            c_idx = (self.Y[:, i] == 1).nonzero()[0].tolist()
            c_num = len(c_idx)
            c_idx = [idx for idx in c_idx if idx not in train_idx]
            random.shuffle(c_idx)
            split_index = int(c_num * ratio) - np.sum(self.Y[train_idx], axis=0)[i]
            if split_index > 0:
                train_idx = train_idx + c_idx[:split_index]
        test_idx = [idx for idx in range(self.n) if idx not in train_idx]
        self.train_idx = list(set(train_idx))
        self.test_idx = list(set(test_idx))

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key in ["device", "X", "H"] and "init" in self.__dict__.keys():
            self.update()


def load_data(data_dir, cfg):
    path_dir, name = osp.split(data_dir)
    if name.lower() == "twitter":
        dataset = load_twitter_data(data_dir, cfg)
        dataset.split_data(cfg["split ratio"])
    else:
        dataset = load_citation_data(data_dir, cfg)
        dataset.split_data(cfg["split ratio"])
    dataset.name = name
    return dataset


def load_citation_data(path_dir, cfg):
    embed_file_path = osp.join(path_dir, "features.txt")
    X = np.loadtxt(embed_file_path)
    label_file_path = osp.join(path_dir, "labels.txt")
    Y = np.loadtxt(label_file_path).astype(np.int32)
    hyperedge_file_path = osp.join(path_dir, "hyperedges.txt")
    H = np.loadtxt(hyperedge_file_path).astype(np.int32)
    assert X.shape[0] == H.shape[1] == Y.shape[0], "The dim does not match"

    dataset = hypergraph_data(X, Y, H)
    return dataset


def load_twitter_data(path_dir, cfg):
    hyperedge_file_path = osp.join(path_dir, "hyperedges.txt")
    H = np.loadtxt(hyperedge_file_path).astype(np.int32)

    embed_file_path = osp.join(path_dir, f"community_id_embed_{cfg['feature type']}.txt")
    X = np.loadtxt(embed_file_path)
    X = X[:, 1:]
    # # load labels
    label_file_path = osp.join(path_dir, "label_comm.txt")
    Y = np.loadtxt(label_file_path).astype(np.int32)
    empty_row = (H.sum(0) < 3).nonzero()[0]
    empty_col = np.all(np.isclose(H.T, 0), axis=1).nonzero()

    H = np.delete(H, empty_row, axis=0)
    H = np.delete(H, empty_col, axis=1)

    X = np.delete(X, empty_col, axis=0)
    Y = np.delete(Y, empty_col, axis=0)

    assert X.shape[0] == H.shape[1] == Y.shape[0], "The dim does not match"
    dataset = hypergraph_data(X, Y, H)

    return dataset
