import random, os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, jaccard_score
import os.path as osp

def get_f1_score(y_pred, y_true):
    assert y_true.shape == y_pred.shape, "The shape of true labels and pred labels does not match"
    return f1_score(y_true=y_true, y_pred=y_pred, average="micro")


def get_jaccard_score(y_pred, y_true):
    assert y_true.shape == y_pred.shape, "The shape of true labels and pred labels does not match"
    return jaccard_score(y_true=y_true, y_pred=y_pred, average="micro")


def eval_scores(y_pred, y_true):
    f1 = get_f1_score(y_pred=y_pred, y_true=y_true)
    jaccard = get_jaccard_score(y_pred=y_pred, y_true=y_true)
    return {"F1": f1, "Jaccard": jaccard}


def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 6
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            print(f'\nRun {run + 1:02d}:')
            print(f'Highest Train F1: {result[:, 0].max():.2f}')
            print(f'Highest Train Jaccard: {result[:, 1].max():.2f}')
            print(f'Highest Valid F1: {result[:, 2].max():.2f}')
            print(f'Highest Valid Jaccard: {result[:, 3].max():.2f}')
            print(f'  Final Train F1: {result[result[:, 2].argmax().item(), 0]:.2f}')
            print(f'  Final Train Jaccard: {result[result[:, 2].argmax().item(), 1]:.2f}')
            print(f'   Final Test F1: {result[result[:, 2].argmax().item(), 5]:.2f}')
            print(f'   Final Test Jaccard: {result[result[:, 2].argmax().item(), 6]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            best_epoch = []
            for r in result:
                index = np.argmax(r[:, 3]).item()
                best_epoch.append(index)
                train_f1 = r[:, 0].max().item()
                train_jaccard = r[:, 1].max().item()
                valid_f1 = r[index, 2].item()
                valid_jaccard = r[index, 3].item()
                best_train_f1 = r[index, 0].item()
                best_train_jaccard = r[index, 1].item()
                test_f1 = r[index, 4].item()
                test_jaccard = r[index, 5].item()

                best_results.append((train_f1, train_jaccard, valid_f1, valid_jaccard,
                                     best_train_f1, best_train_jaccard, test_f1, test_jaccard))

            best_result = torch.tensor(best_results)

            # print(f'All runs:')
            # print("best epoch:", best_epoch)
            # r = best_result[:, 0]
            # print(f'Best Train F1: {r.mean():.2f} ± {r.std():.2f}')
            # r = best_result[:, 1]
            # print(f'Best Train Jaccard: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid F1: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'Highest Valid Jaccard: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'Highest Train F1: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 5]
            print(f'Highest Train Jaccard: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 6]
            print(f'Highest Test F1: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 7]
            print(f'Highest Test Jaccard: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 2], best_result[:, 3], best_result[:, 6], best_result[:, 7]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            #             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


def load_yaml(file_dir, dname):
    file_path = osp.join(file_dir, f"{dname}.yaml")
    with open(file_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data


def rand_train_test_idx(Y, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    train_idx, valid_idx, test_idx = [], [], []
    num_classes = Y.shape[1]
    for i in range(num_classes):
        c_idx = (Y[:, i] == 1).nonzero()[0].tolist()
        c_num = len(c_idx)
        c_idx = [idx for idx in c_idx if idx not in train_idx]
        random.shuffle(c_idx)
        split_idx = int(c_num * train_prop) - np.sum(Y[train_idx], axis=0)[i]
        if split_idx > 0:
            train_idx = train_idx + c_idx[:split_idx]
        c_idx = [idx for idx in c_idx[split_idx:] if idx not in valid_idx]
        random.shuffle(c_idx)
        split_idx = int(c_num * valid_prop) - np.sum(Y[valid_idx], axis=0)[i]
        if split_idx > 0:
            valid_idx = valid_idx + c_idx[:split_idx]
    test_idx = [idx for idx in range(Y.shape[0]) if idx not in train_idx and idx not in valid_idx]

    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def evaluate(y_true, y_pred, split_idx, num_classes, loss_function):
    y_true, y_pred = y_true.cpu(), y_pred.cpu()

    if len(y_pred.shape) == 1 and len(y_true.shape) == 1:
        y_pred = F.one_hot(y_pred, num_classes=num_classes)
        y_true = F.one_hot(y_true, num_classes=num_classes)
    # sum_y_pred = y_pred.sum(0)
    # sum_y_true = y_true.sum(0)
    # index_y_pred = torch.argsort(sum_y_pred)
    # index_y_true = torch.argsort(sum_y_true)
    # y_pred = y_pred[:, index_y_pred]
    # y_true = y_true[:, index_y_true]

    train_result = eval_scores(y_pred=y_pred[split_idx['train']], y_true=y_true[split_idx['train']])
    valid_result = eval_scores(y_pred=y_pred[split_idx['valid']], y_true=y_true[split_idx['valid']])
    test_result = eval_scores(y_pred=y_pred[split_idx['test']], y_true=y_true[split_idx['test']])
    train_f1, train_jaccard = train_result["F1"], train_result["Jaccard"]
    valid_f1, valid_jaccard = valid_result["F1"], valid_result["Jaccard"]
    test_f1, test_jaccard = test_result["F1"], test_result["Jaccard"]

    return train_f1, train_jaccard, valid_f1, valid_jaccard, test_f1, test_jaccard
