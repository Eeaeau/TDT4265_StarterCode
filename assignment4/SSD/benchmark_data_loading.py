import click
import numpy as np
import time
import torch
from tops.config import instantiate, LazyConfig
from tops import to_cuda
from pathlib import Path
np.random.seed(0)


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(config_path):
    cfg = LazyConfig.load(str(config_path))
    dataloader = instantiate(cfg.data_train.dataloader)
    gpu_transform = instantiate(cfg.data_train.gpu_transform)
    for batch in dataloader: # Warmup
        batch = to_cuda(batch)
        batch = gpu_transform(batch)
    torch.cuda.synchronize()
    start_time = time.time()
    n_images = 0
    for batch in dataloader:
        batch = to_cuda(batch)
        batch = gpu_transform(batch)
        n_images += batch["image"].shape[0]
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    images_per_sec = n_images / elapsed_time
    print("Data pipeline runtime:", images_per_sec, "images/sec")


if __name__ == "__main__":
    main()