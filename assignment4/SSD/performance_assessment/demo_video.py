import pathlib
import torch
import tqdm
import click
import numpy as np
import cv2
import tops
from ssd import utils
from tops.config import instantiate
from PIL import Image
from vizer.draw import draw_boxes
from tops.checkpointer import load_checkpoint
from pathlib import Path

@torch.no_grad()
@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.argument("video_path", type=click.Path(dir_okay=True, path_type=str))
@click.argument("output_path", type=click.Path(dir_okay=True, path_type=str))
@click.option("-s", "--score_threshold", type=click.FloatRange(min=0, max=1), default=.5)
def run_demo(config_path: str, score_threshold: float, video_path: str, output_path: str):
    cfg = utils.load_config(config_path)
    model = tops.to_cuda(instantiate(cfg.model))
    model.eval()
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])
    width, height = 1024, 128

    reader = cv2.VideoCapture(video_path) 
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    cpu_transform = instantiate(cfg.data_val.dataset.transform)
    gpu_transform = instantiate(cfg.data_val.gpu_transform)
    video_length = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    assert reader.isOpened()
    for frame_idx in tqdm.trange(video_length, desc="Predicting on video"):
        ret, frame = reader.read()
        assert ret, "An error occurred"
        frame = np.ascontiguousarray(frame[:, :, ::-1])
        img = cpu_transform({"image": frame})["image"].unsqueeze(0)
        img = tops.to_cuda(img)
        img = gpu_transform({"image": img})["image"]
        boxes, categories, scores = model(img, score_threshold=score_threshold)[0]
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes, categories, scores = [_.cpu().numpy() for _ in [boxes, categories, scores]]
        frame = draw_boxes(
            frame, boxes, categories, scores).astype(np.uint8)
        writer.write(frame[:, :, ::-1])
    print("Video saved to:", pathlib.Path(output_path).absolute())
        
if __name__ == '__main__':
    run_demo()