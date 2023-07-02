# -*- coding:utf-8 -*-
from .config import get_config, save_config
import yaml, os

def join(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.sep.join(seq)

yaml.add_constructor('!join', join)