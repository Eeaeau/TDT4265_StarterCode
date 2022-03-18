import os
import torch
import pathlib
from ..logger import global_step, log, logger
from typing import List, Optional
from argparse import ArgumentError

_checkpoint_dir = None
_models = None

def init(checkpoint_dir: pathlib.Path):
    global _checkpoint_dir
    _checkpoint_dir = checkpoint_dir


def load_checkpoint(
        checkpoint_path: Optional[os.PathLike] = None,
        load_best: bool = False,
        map_location=None) -> dict:
    if map_location is None:
        map_location = f"cuda:0" if torch.cuda.is_available() else "cpu"
    if checkpoint_path is None:
        checkpoint_path = _checkpoint_dir
        if _checkpoint_dir is None:
            raise ArgumentError(
                "Both the provided checkpoint_path and global checkpoint_dir is None." +
                "You have to initialize mlops or provide a checkpoint.")
    checkpoint_path = pathlib.Path(checkpoint_path)
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        log(f"Loaded checkpoint from {checkpoint_path}")
        return ckpt
    checkpoint_dir = checkpoint_path
    if load_best:
        checkpoint_path = checkpoint_dir.joinpath("best_model.ckpt")
    else:
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"No checkpoint folder exists in: {checkpoint_dir.absolute()}")
        
        checkpoints = get_ckpt_paths(checkpoint_dir)
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No checkpoints in folder: {checkpoint_path}")
        checkpoint_path = checkpoints[-1]
    assert checkpoint_path.is_file(), "This should not be reachable."
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Did not find file: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    log(f"Loaded checkpoint from {checkpoint_path}")
    return ckpt

def get_ckpt_paths(checkpoint_dir: pathlib.Path) -> List[pathlib.Path]:
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoints = [x for x in checkpoint_dir.glob("*.ckpt") if x.stem != "best_model"]
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return checkpoints


def save_checkpoint(
        state_dict: dict,
        checkpoint_dir: Optional[os.PathLike] = None,
        is_best: bool = False,
        max_keep=1) -> None:
    """
    Args:
        checkpoint_path: path to file or folder.
    """
    if checkpoint_dir is None:
        assert _checkpoint_dir is not None
        checkpoint_dir = _checkpoint_dir
    checkpoint_dir.parent.mkdir(exist_ok=True, parents=True)
    previous_checkpoint_paths = get_ckpt_paths(checkpoint_dir)
    if is_best:
        torch.save(state_dict, checkpoint_dir.joinpath("best_model.ckpt"))
        log(f"Saved model to: {checkpoint_dir.joinpath('best_model.ckpt')}" )
    checkpoint_path = checkpoint_dir.joinpath(f"{global_step()}.ckpt")
    if checkpoint_path.is_file():
        return
    torch.save(state_dict, checkpoint_path)
    log(f"Saved model to: {checkpoint_path}")
    if len(previous_checkpoint_paths) > max_keep:
        previous_checkpoint_paths[0].unlink()


def has_checkpoint(checkpoint_dir: Optional[os.PathLike] = None) -> bool:
    if checkpoint_dir is None:
        assert _checkpoint_dir is not None
        checkpoint_dir = _checkpoint_dir
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    num_checkpoints = len(list(checkpoint_dir.glob("*.ckpt")))
    return num_checkpoints > 0


def register_models(models: dict):
    global _models
    for key, model in models.items():
        if not hasattr(model, "state_dict"):
            raise ArgumentError("The model has to have a state_dict")
        if not hasattr(model, "load_state_dict"):
            raise ArgumentError("The model has to have load_state_dict")
    _models = models


def save_registered_models(other_state: dict = None, **kwargs):
    assert _models is not None
    state_dict = {key: model.state_dict() for key, model in _models.items()}
    if other_state:
        assert all(key not in state_dict for key in other_state)
        state_dict.update(other_state)
    save_checkpoint(state_dict, **kwargs)
    logger._write_metadata()

def load_registered_models(**kwargs):
    assert _models is not None
    state_dict = load_checkpoint(**kwargs)
    for key, state in state_dict.items():
        if key in _models:
            _models[key].load_state_dict(state)
    return {k: v for k,v in state_dict.items() if key not in _models}