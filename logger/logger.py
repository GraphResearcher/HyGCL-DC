# -*- coding:utf-8 -*-
import logging
import sys

def get_logger(file_path, hdl):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if "stdout" in hdl:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        logger.addHandler(stdout_handler)
    if "file" in hdl:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger
