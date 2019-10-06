try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


import os
import numpy as np
# import matplotlib.pyplot as plt
import pickle
import math
import csv
from sklearn.manifold import TSNE
import gzip
import pickle
import datetime
from queue import Queue


exists = os.path.exists
join = os.path.join


def curr_time_str():
    return datetime.datetime.now().strftime('%y%m%d%H%M%S')


def mkdir(dir):
    if not exists(dir):
        os.mkdir(dir)
    return dir


def aspath(path):
    assert exists(path)
    return path


def traverse(folder, postfix='', rec=False, only_file=True):
    """
    Traverse all files in the given folder
    :param folder: The name of the folder to traverse.
    :param postfix: Required postfix
    :param rec: recursively or not
    :param only_file: Do not yield folder
    :return: paths of the required files
    """
    q = Queue()
    q.put(aspath(folder))
    while not q.empty():
        folder = q.get()
        for path in os.listdir(folder):
            path = join(folder, path)
            if os.path.isdir(path):
                q.put(path)
                if only_file:
                    continue
            if path.endswith(postfix):
                yield path
        if not rec:
            break
