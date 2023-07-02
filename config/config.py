# -*- coding:utf-8 -*-
import yaml


def get_config(path="config/config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def save_config(cfg, path="config/config.yaml"):
    with open(path, "w") as f:
        yaml.dump(cfg, stream=f, default_flow_style=False, sort_keys=False)

