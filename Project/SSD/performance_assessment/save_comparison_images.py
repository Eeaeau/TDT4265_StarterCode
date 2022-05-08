import cv2
import os

import torch
import tops
import click
import numpy as np
from tops.config import instantiate
from tops.config import LazyCall as L
from tops.checkpointer import load_checkpoint
from tops.misc import (
    get_config,
    get_trained_model,
    get_dataloader,
    convert_boxes_coords_to_pixel_coords,
    convert_image_to_hwc_byte,
    create_filepath,
    get_save_folder_name
)

from vizer.draw import draw_boxes
from ssd import utils
from tqdm import tqdm
from ssd.data.transforms import ToTensor
from visualize_model_cam import get_cam_image
import torch

def visualize_annotations_on_image(image, batch, label_map):
    boxes = convert_boxes_coords_to_pixel_coords(batch["boxes"][0], batch["width"], batch["height"])
    labels = batch["labels"][0].cpu().numpy().tolist()

    image_with_boxes = draw_boxes(image, boxes, labels, class_name_map=label_map)
    return image_with_boxes


def visualize_model_predictions_on_image(image, img_transform, batch, model, label_map, score_threshold):
    pred_image = tops.to_cuda(batch["image"])
    transformed_image = img_transform({"image": pred_image})["image"]

    boxes, categories, scores = model(transformed_image, score_threshold=score_threshold)[0]
    boxes = convert_boxes_coords_to_pixel_coords(boxes.detach().cpu(), batch["width"], batch["height"])
    categories = categories.cpu().numpy().tolist()

    image_with_predicted_boxes = draw_boxes(image, boxes, categories, scores, class_name_map=label_map)

    return image_with_predicted_boxes


def create_comparison_image(batch, model, img_transform, label_map, score_threshold, cam_enabled):
    image = batch["image"]

    image_8bit = convert_image_to_hwc_byte(image)

    image_with_annotations = visualize_annotations_on_image(image_8bit, batch, label_map)
    # TODO: add cam here to visualize the model predictions
    # print("Image shape:", image.shape, "Image type:", type(image), image.dtype)
    if cam_enabled:
        try:
            pred_image = tops.to_cuda(batch["image"])
            transformed_image = img_transform({"image": pred_image})["image"]
            boxes, categories, scores = model(transformed_image, score_threshold=score_threshold)[0]
            boxes = convert_boxes_coords_to_pixel_coords(boxes.detach().cpu(), batch["width"], batch["height"])
            image = (get_cam_image(model=model, input_tensor=image, labels=label_map, boxes=boxes, norm=False, renorm=True)).astype(np.uint8)
        except:
            print("Could not visualize model predictions on image")
            image = convert_image_to_hwc_byte(image)
        # print("Boxes shape:", boxes.shape, "Boxes type:", type(boxes), boxes)

    else:
        image = convert_image_to_hwc_byte(image)

    image_with_model_predictions = visualize_model_predictions_on_image(
        image, img_transform, batch, model, label_map, score_threshold)

    concatinated_image = np.concatenate([
        image_8bit,
        image_with_annotations,
        image_with_model_predictions
    ], axis=0)
    return concatinated_image


def create_and_save_comparison_images(dataloader, model, cfg, save_folder, score_threshold, num_images, cam_enabled):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Saving images to", save_folder)

    num_images_to_save = min(len(dataloader), num_images)
    dataloader = iter(dataloader)

    img_transform = instantiate(cfg.data_val.gpu_transform)
    for i in tqdm(range(num_images_to_save)):
        batch = next(dataloader)
        comparison_image = create_comparison_image(batch, model, img_transform, cfg.label_map, score_threshold, cam_enabled)
        filepath = create_filepath(save_folder, i)
        cv2.imwrite(filepath, comparison_image[:, :, ::-1])

@click.command()
@click.argument("config_path")
@click.option("--train", default=False, is_flag=True, help="Use the train dataset instead of val")
@click.option("-n", "--num_images", default=500, type=int, help="The max number of images to save")
@click.option("-c", "--conf_threshold", default=0.3, type=float, help="The confidence threshold for predictions")
@click.option("-cam", "--cam_enabled", default=False, type=bool, help="Visualize the model cam")
def main(config_path, train, num_images, conf_threshold, cam_enabled):
    cfg = get_config(config_path)
    model = get_trained_model(cfg)

    if train:
        dataset_to_visualize = "train"
    else:
        dataset_to_visualize = "val"

    dataloader = get_dataloader(cfg, dataset_to_visualize)
    save_folder = get_save_folder_name(cfg, dataset_to_visualize)

    create_and_save_comparison_images(dataloader, model, cfg, save_folder, conf_threshold, num_images, cam_enabled)


if __name__ == '__main__':
    main()
