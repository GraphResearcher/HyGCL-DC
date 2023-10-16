# -*- coding:utf-8 -*-
import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable


class HyperGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, a, b, reapproximate=True, cuda=True):
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m).to(self.cuda)
        else:
            A = structure

        if self.cuda: A = A.to(self.cuda)
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)
        return AHW + b

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


class HyperGCN(nn.Module):
    def __init__(self, num_features, num_layers, num_classes, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()

        self.num_features = num_features
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_hidden = args.num_hidden
        self.reapproximate = not args.HyperGCN_fast
        self.MLP_hidden = args.MLP_hidden

        self.dropout = args.dropout
        self.num_layers = num_layers
        self.mediators = args.HyperGCN_mediators

        if args.cuda in [0, 1, 2, 3]:
            device = torch.device('cuda:' + str(args.cuda)
                                  if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')

        h = [self.num_features]
        for _ in range(self.num_layers):
            h.append(self.num_hidden)

        self.layers = nn.ModuleList(
            [HyperGraphConvolution(h[i], h[i + 1], self.reapproximate, device) for i in range(self.num_layers)])

        self.net = MLP(in_channels=self.num_hidden, hidden_channels=self.MLP_hidden, out_channels=num_classes,
                       num_layers=1, dropout=self.dropout)
        # self.proj_head = MLP(in_channels=args.MLP_hidden,hidden_channels=args.MLP_hidden,out_channels=args.MLP_hidden, num_layers=1,
        #                      dropout=self.dropout)
        self.proj_head = nn.Linear(self.num_hidden,self.num_hidden)
        self.structure=None

    def reset_parameters(self):
        self.net.reset_parameters()
        self.proj_head.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data, structure=None):
        Z = data.X_TORCH
        if structure == None:
            structure = self.structure
        for i, hidden in enumerate(self.layers):
            Z = F.relu(hidden(structure, Z, self.mediators))
            Z = F.dropout(Z, self.dropout, training=self.training)
        Z = self.net(Z)
        return Z

    def forward_cl(self, data, structure=None):
        pgd1 = torch.rand_like(data.X_TORCH)
        Z = data.X_TORCH + pgd1
        # Z = data.X_TORCH
        if structure == None:
            structure = self.structure
        for i, hidden in enumerate(self.layers):
            Z = F.relu(hidden(structure, Z, self.mediators))
            if i < self.num_layers - 1:
                Z = F.dropout(Z, self.dropout, training=self.training)
        Z = self.proj_head(Z)
        return Z


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if normalization.__class__.__name__ != 'Identity':
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            # x = F.relu(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns:
    updated data with 'graph' as a key and its value the approximated hypergraph
    """

    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for k in E:
        hyperedge = k.nonzero()[0]
        p = np.dot(X[hyperedge], rv)  # projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2 * len(hyperedge) - 3  # normalisation constant
        if m:
            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])

            if (Se, Ie) not in weights:
                weights[(Se, Ie)] = 0
            weights[(Se, Ie)] += float(1 / c)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / c)

            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se, mediator], [Ie, mediator], [mediator, Se], [mediator, Ie]])
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se, Ie], [Ie, Se]])
            e = len(hyperedge)

            if (Se, Ie) not in weights:
                weights[(Se, Ie)] = 0
            weights[(Se, Ie)] += float(1 / e)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / e)

    return adjacency(edges, weights, V)


def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """

    if (Se, mediator) not in weights:
        weights[(Se, mediator)] = 0
    weights[(Se, mediator)] += float(1 / c)

    if (Ie, mediator) not in weights:
        weights[(Ie, mediator)] = 0
    weights[(Ie, mediator)] += float(1 / c)

    if (mediator, Se) not in weights:
        weights[(mediator, Se)] = 0
    weights[(mediator, Se)] += float(1 / c)

    if (mediator, Ie) not in weights:
        weights[(mediator, Ie)] = 0
    weights[(mediator, Ie)] += float(1 / c)

    return weights


def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """

    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]
    organised = []

    for e in edges:
        i, j = e[0], e[1]
        w = weights[(i, j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A


def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.
    dhi[np.isnan(dhi)] = 0.
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)


def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """

    M = M.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)

    return torch.sparse.FloatTensor(indices, values, shape)