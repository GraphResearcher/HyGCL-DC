from config import get_config
import scipy.sparse as sp
import os
import os.path as osp
from utils import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model import HyperGCN, Laplacian
from logger.logger import get_logger
from datetime import datetime
from itertools import combinations

cfg = get_config()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "True" if cfg["cuda"] in [0, 1, 2, 3] else "False"
os.environ['PYTHONHASHSEED'] = str(cfg["seed"])

if cfg["cuda"] in [0, 1, 2, 3]:
    device = torch.device('cuda:' + str(cfg["cuda"])
                          if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')


def parse_model(dataset, cfg):
    n_layers = cfg["n_layers"]
    n_hid = cfg["hidden dim"]
    V, XE, L = dataset.n, dataset.XE, dataset.L
    in_ch, n_classes = dataset.n_features, dataset.n_cls
    mlphid = cfg["cl hidden dim"]
    dropout = cfg["dropout"]
    cuda = type(cfg["cuda"]) is int
    if cfg["model"] == "HyGCL-DC":
        reapproximate = False
        mediators = True
        structure = Laplacian(V, L, XE, True)
        model = HyperGCN(in_ch, n_layers, n_hid, n_classes, structure, reapproximate, dropout, mediators,
                         mlphid, cuda)
    return model


def train(model, dataset, cfg, device, cl=False):
    V, L = dataset.n, dataset.L
    X, XE = dataset.X_TORCH, dataset.XE_TORCH
    if cfg["model"] == "HyGCL-DC":
        if cl:
            from model import Laplacian
            structure = Laplacian(V, L, XE.cpu(), True).to(device)
            out = model.forward_cl(XE, structure)
        else:
            out = model.forward(XE)
    return out


def get_evaluation_metric(y_pred, y_true, n_cls):
    if len(y_pred.shape) == 1:
        y_pred = F.one_hot(y_pred, num_classes=n_cls)
    if len(y_true.shape) == 1:
        y_true = F.one_hot(y_true, num_classes=n_cls)
    sum_y_pred = y_pred.sum(0)
    sum_y_true = y_true.sum(0)
    index_y_pred = torch.argsort(sum_y_pred)
    index_y_true = torch.argsort(sum_y_true)
    y_pred = y_pred[:, index_y_pred]
    y_true = y_true[:, index_y_true]
    return eval_scores(y_pred, y_true)


def run(cfg, aug_method=["mask", "hyperedge"]):
    best = 0.0
    best_result = None
    dataset_dir = osp.join(cfg["data_dir"], cfg["dataset"])
    dataset = load_data(dataset_dir, cfg)

    dataset.device = device
    dataset.update()

    X, Y = dataset.X_TORCH, dataset.Y_TORCH
    overlapping = dataset.name == "Twitter"

    Y = torch.where(Y)[1] if not overlapping else Y

    train_idx, test_idx = dataset.train_idx, dataset.test_idx

    model = parse_model(dataset, cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    activation, loss_function, cl_function, pred_function = get_functions(cfg["dataset"])
    model = model.to(device)

    y_true = Y
    pbar = tqdm(range(cfg["epoch"]))

    for epoch in pbar:

        model.train()
        optimizer.zero_grad()

        cl_loss, model_loss, loss = 0.0, 0.0, 0.0

        if aug_method:
            # data_aug1 = aug(dataset, ["mask", "hyperedge"], True)
            # data_aug2 = aug(dataset, ["mask", "hyperedge"], True)

            data_aug1 = aug(dataset, aug_method, True)
            data_aug2 = aug(dataset, aug_method, True)

            out1 = train(model=model, cl=True, cfg=cfg, dataset=data_aug1, device=device)
            out2 = train(model=model, cl=True, cfg=cfg, dataset=data_aug2, device=device)

            cl_loss = cl_function(out1, out2, cfg["cl temperature"])

            loss = cl_loss * cfg["alpha"]

        output = train(model=model, cl=False, cfg=cfg, dataset=dataset, device=device)
        y_pred = activation(output) if activation == F.sigmoid else activation(output, dim=1)

        model_loss = loss_function(y_pred[train_idx], y_true[train_idx])
        loss += model_loss

        loss.backward()
        optimizer.step()

        if epoch % cfg["print epoch"] == 0:
            model.eval()
            output = train(model=model, cl=False, cfg=cfg, dataset=dataset, device=device)
            y_pred = activation(output) if activation == F.sigmoid else activation(output, dim=1)
            y_pred = pred_function(y_pred.detach(), cfg["cl threshold"]).type_as(Y).cpu()
            result = get_evaluation_metric(y_pred=y_pred[test_idx], y_true=y_true.cpu()[test_idx], n_cls=dataset.n_cls)
            measure = result["f1 score"]
            if measure > best:
                best = measure
                best_result = result
            pbar.set_description(f' model Loss: {model_loss:.4f} '
                                 f'cl loss: {cl_loss:.4f}  total loss: {loss:.4f} best: {best:.4f}')

    return best_result


def run5(cfg, aug_method):
    f1, jaccard, mod, nmi = [], [], [], []
    for idx in range(5):
        result = run(cfg, aug_method=aug_method)
        print(result)
        f1.append(result["f1 score"])
        jaccard.append(result["Jaccard"])

        # curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # np.savetxt(rf"D:\Program\CommunityDetection\ablation\TSNE\{cfg['dataset']}_{cfg['model']}_{curr_time}.txt", data)

    return {"f1 score": np.round(np.mean(f1) * 100, 2), "f1 score std": np.round(np.std(f1) * 100, 2),
            "jaccard": np.round(100 * np.mean(jaccard), 2), "jaccard std": np.round(np.std(jaccard) * 100, 2)}


if __name__ == '__main__':
    aug_method = ["mask", "hyperedge"]
    log_file = "train5.log"
    logger = get_logger(osp.join(cfg["logger_dir"], log_file), hdl=["file", "stdout"])
    result = run5(cfg, aug_method=aug_method)
    logger.info({"model": cfg["model"], "dataset": cfg["dataset"]})
    logger.info(result)
