import os
import sys
import logging
from os import makedirs
from os.path import dirname

from multiprocessing import Lock, Pool


def add_stdout_logger():
    """Setup stdout logging"""

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S', )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def ensure_dir_exists(filename):
    """Make sure the parent directory of `filename` exists"""
    makedirs(dirname(filename), exist_ok=True)

def make_dir(output_dir):
    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) > 0:
            logging.warning("Output dir exists and is non-empty")
    else:
        os.makedirs(output_dir)
