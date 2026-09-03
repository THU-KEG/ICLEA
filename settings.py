import os
import random
from os.path import abspath, dirname, join

import numpy as np
import torch

def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
PROJ_DIR = abspath(dirname(__file__))
DATA_DIR = join(PROJ_DIR, 'data')
OUT_DIR = join(PROJ_DIR, 'out')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB_SIZE = 100000
LaBSE_DIM = 768
DESC_DIM = 768
BATCH_SIZE = 64
NEIGHBOR_SIZE = 15
MULTI_HEAD_DIM = 1
