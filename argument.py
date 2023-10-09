import argparse
import os
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser()
    root_dir = os.getcwd()
    parser.add_argument("--root_dir", type=str, default=root_dir)
    parser.add_argument("--data_dir", type=str, default=osp.join(root_dir, "data"))
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--train_prop', type=float, default=0.6)
    parser.add_argument('--valid_prop', type=float, default=0.2)
    parser.add_argument('--dname', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.001, type=float)

    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument("--num_hidden", default=256, type=int)
    parser.add_argument('--MLP_hidden', default=256, type=int)
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder

    parser.add_argument("--aug_ratio", type=float)
    parser.add_argument('--aug_method', type=str)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--cl_temperature', default=0.6, type=float)
    parser.add_argument('--HyperGCN_fast', default=True, type=bool)
    parser.add_argument('--HyperGCN_mediators', default=True, type=bool)

    parser.add_argument('--display_step', type=int, default=1)

    parser.set_defaults(dname="citeseer")
    parser.set_defaults(method="HyGCL-DC")
    parser.set_defaults(aug_method="mask")
    parser.set_defaults(threshold=0.45)
    parser.set_defaults(aug_ratio=0.2)

    args = parser.parse_args()
    return args