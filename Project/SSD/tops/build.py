from os import PathLike
from typing import Optional
from .logger.logger import init as _init_logger
from .checkpointer.checkpointer import init as init_checkpointer
from pathlib import Path
from .utils.git_diff import dump_git_diff


def init(
        output_dir,
        logging_backend=["stdout", "json", "tensorboard"],
        checkpoint_dir: Optional[PathLike] = None
    ):
    """
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    _init_logger(output_dir.joinpath("logs"), logging_backend)
    if checkpoint_dir is None:
        checkpoint_dir = output_dir.joinpath("checkpoints")
    init_checkpointer(checkpoint_dir)
    dump_git_diff(output_dir)



