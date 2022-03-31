import torch
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
from tops.config import LazyConfig
from os import PathLike

def batch_collate(batch):
    elem = batch[0]
    batch_ = {key: default_collate([d[key] for d in batch]) for key in elem}
    return batch_

def batch_collate_val(batch):
    """
        Same as batch_collate, but removes boxes/labels from dataloader
    """
    elem = batch[0]
    ignore_keys = set(("boxes", "labels"))
    batch_ = {key: default_collate([d[key] for d in batch]) for key in elem if key not in ignore_keys}
    return batch_


def class_id_to_name(labels, label_map: list):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().tolist()
    return [label_map[idx] for idx in labels]


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]


def load_config(config_path: PathLike):
    config_path = Path(config_path)
    run_name = "_".join(config_path.parts[1:-1]) + "_" + config_path.stem
    cfg = LazyConfig.load(str(config_path))
    cfg.output_dir = Path(cfg.train._output_dir).joinpath(*config_path.parts[1:-1], config_path.stem)
    cfg.run_name = run_name
    return cfg
