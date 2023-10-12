# -*- coding:utf-8 -*-
import torch
import numpy as np
from itertools import combinations
import numpy.random as random
import os.path as osp
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from model import Laplacian


class hypergraph_data():
    def __init__(self, X, Y, H, device=None):
        self.X, self.Y, self.H = X, Y, H
        self.overlapping = False
        self.device = device

        A = Laplacian(self.X.shape[0], self.H, self.X, True).coalesce()
        #  Add self loop

        N = self.X.shape[0]
        eye_values = torch.ones(N).float()
        eye_indices = torch.arange(0, N).long().unsqueeze(0).repeat(2, 1)

        indices = torch.cat((A.indices(), eye_indices), dim=1)
        values = torch.cat((A.values(), eye_values))
        shape = A.shape

        self.structure = torch.sparse.FloatTensor(indices, values, shape).coalesce().to(self.device)

        self.dname = None
    def update(self):
        self.Y_TORCH = torch.FloatTensor(self.Y).to(self.device)
        if self.overlapping==False:
            self.Y_TORCH = self.Y_TORCH.nonzero()[:,1]
        self.X_TORCH = torch.FloatTensor(self.X).to(self.device)
        self.H_TORCH = torch.FloatTensor(self.H).to(self.device)



    def print(self):
        print(f"Dataset: {self.dname}, N: {self.X.shape[0]}, M: {self.H.shape[0]}, C: {self.Y.shape[1]}, d: {self.X.shape[1]}")

def load_data(args):
    data_dir, dname = args.data_dir, args.dname
    data_path = osp.join(data_dir, dname)
    if dname.lower() == "twitter":
        dataset = load_twitter_data(data_path)
        dataset.overlapping = True
    else:
        dataset = load_citation_data(data_path)
    args.num_classes = dataset.Y.shape[1]
    args.num_features = dataset.X.shape[1]
    dataset.dname = dname
    dataset.print()
    return dataset, args


def load_citation_data(path_dir):
    embed_file_path = osp.join(path_dir, "features.txt")
    X = np.loadtxt(embed_file_path)
    label_file_path = osp.join(path_dir, "labels.txt")
    Y = np.loadtxt(label_file_path).astype(np.int32)
    hyperedge_file_path = osp.join(path_dir, "hyperedges.txt")
    H = np.loadtxt(hyperedge_file_path).astype(np.int32)
    assert X.shape[0] == H.shape[1] == Y.shape[0], "The dim does not match"

    dataset = hypergraph_data(X, Y, H)
    return dataset


def load_twitter_data(path_dir):
    hyperedge_file_path = osp.join(path_dir, "hyperedges.npz")
    H = load_npz(hyperedge_file_path).astype(np.int32).toarray()

    embed_file_path = osp.join(path_dir, f"features.txt")
    X = np.loadtxt(embed_file_path)
    X = X[:, 1:]
    # # load labels
    label_file_path = osp.join(path_dir, "labels.txt")
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