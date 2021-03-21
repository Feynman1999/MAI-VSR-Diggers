import random
from functools import partial
import numpy as np
from edit.utils import build_from_cfg
from .dataset_wrappers import RepeatDataset
from .registry import DATASETS


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    It supports a variety of dataset config. If ``cfg`` is a Sequential (list
    or dict), it will be a concatenated dataset of the datasets specified by
    the Sequential. If it is a ``RepeatDataset``, then it will repeat the
    dataset ``cfg['dataset']`` for ``cfg['times']`` times. If the ``ann_file``
    of the dataset is a Sequential, then it will build a concatenated dataset
    with the same dataset type but different ``ann_file``.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    if isinstance(cfg, (list, tuple)):
        raise NotImplementedError("dose not support list(tuple) configs for dataset build now")
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        raise NotImplementedError("does not support list(tuple) ann_files for dataset build now")
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
