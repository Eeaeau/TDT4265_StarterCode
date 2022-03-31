import random
import numpy as np
import torch

AMP_enabled = False

def set_AMP(value: bool):
    global AMP_enabled
    AMP_enabled = value


def AMP():
    return AMP_enabled


def _to_cuda(element):
    return element.to(get_device(), non_blocking=True)


def to_cuda(elements):
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [_to_cuda(x) for x in elements]
    if isinstance(elements, dict):
        return {k: _to_cuda(v) for k,v in elements.items()}
    return _to_cuda(elements)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
