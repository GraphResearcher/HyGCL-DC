# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F


def get_cl_loss_function(name):
    if name == "InfoNCE":
        return InfoNCE


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def InfoNCE(z1, z2, T=0.2):
    l1 = semi_loss(z1, z2, T)
    l2 = semi_loss(z2, z1, T)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret


def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def get_loss_function(name):
    if name == "BCEwithLogits":
        return F.binary_cross_entropy_with_logits
    elif name == "BCELOSS":
        return F.binary_cross_entropy
    elif name == "nll_loss":
        return F.nll_loss
    elif name == "MULTILABELSOFTMARGINLOSS":
        return F.multilabel_soft_margin_loss
    elif name == "MULTILABELMARGINLOSS":
        return F.multilabel_margin_loss


def get_activation_function(name):
    if name == "softmax":
        return F.softmax
    elif name == "log_softmax":
        return F.log_softmax
    elif name == "sigmoid":
        return F.sigmoid


def get_pred_function(overlapping):
    if overlapping:
        return overlapping_prediction
    else:
        return classification_prediction


def overlapping_prediction(output, threshold):
    return (output > threshold)


def classification_prediction(output, threshold=None):
    return output.max(1)[1]


def get_functions(dataname):
    if dataname == "Twitter":
        activation = get_activation_function("sigmoid")
        loss_function = get_loss_function("BCELOSS")
        cl_function = get_cl_loss_function("InfoNCE")
        pred_function = get_pred_function(True)
    elif dataname in ["Citeseer-citation", "Cora-author", "Cora-citation"]:
        activation = get_activation_function("log_softmax")
        loss_function = get_loss_function("nll_loss")
        cl_function = get_cl_loss_function("InfoNCE")
        pred_function = get_pred_function(False)
    return activation, loss_function, cl_function, pred_function
