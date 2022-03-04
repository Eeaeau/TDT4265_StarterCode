import time
import click
import torch
import tops
from ssd import utils
from pathlib import Path
from tops.config import instantiate
from tops.checkpointer import load_checkpoint


@torch.no_grad()
def evaluation(cfg, N_images: int):
    model =instantiate(cfg.model)
    model.eval()
    model = tops.to_cuda(model)
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])
    dataloader_val = instantiate(cfg.data_val.dataloader)
    batch = next(iter(dataloader_val))
    gpu_transform = instantiate(cfg.data_val.gpu_transform)
    batch = tops.to_cuda(batch)
    batch = gpu_transform(batch)
    images = batch["image"]
    imshape = list(images.shape[2:])
    # warmup
    print("Checking runtime for image shape:", imshape)
    for i in range(10):
        model(images)
    start_time = time.time()
    for i in range(N_images):
        outputs = model(images)
    total_time = time.time() - start_time
    print("Runtime for image shape:", imshape)
    print("Total runtime:", total_time)
    print("FPS:", N_images / total_time)

@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-n", "--n-images", default=100, type=int)
def main(config_path: Path, n_images: int):
    cfg = utils.load_config(config_path)
    evaluation(cfg, n_images)


if __name__ == '__main__':
    main()