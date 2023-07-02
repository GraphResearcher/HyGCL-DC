# -*- coding:utf-8 -*-

from sklearn.metrics import f1_score, jaccard_score


def get_f1_score(y_pred, y_true):
    assert y_true.shape == y_pred.shape, "The shape of true labels and pred labels does not match"
    return f1_score(y_true=y_true, y_pred=y_pred, average="micro")

def get_jaccard_score(y_pred, y_true):
    assert y_true.shape == y_pred.shape, "The shape of true labels and pred labels does not match"
    return jaccard_score(y_true=y_true, y_pred=y_pred, average="micro")

def eval_scores(y_pred, y_true):
    f1 = get_f1_score(y_pred, y_true)
    jaccard = get_jaccard_score(y_pred, y_true)
    eval_dict = {"f1 score": f1,  "Jaccard": jaccard}
    return eval_dict
