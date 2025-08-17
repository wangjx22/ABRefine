"""Code."""
import os
import random
import yaml
import ml_collections

import numpy as np

import torch


def default_setting(seed, use_cuda=False, deterministic=False):
    """Run default_setting method."""
    # code.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def read_yaml_to_dict(yaml_path):
    """Run read_yaml_to_dict method."""
    # code.
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)
    return ml_collections.ConfigDict(hyp)
