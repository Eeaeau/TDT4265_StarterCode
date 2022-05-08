from email.policy import default
from statistics import mode
import sys, os
sys.path.append(os.path.dirname(os.getcwd())) # Include ../SSD in path
import numpy as np
import torch
import matplotlib.pyplot as plt
from vizer.draw import draw_boxes
from tops.config import instantiate, LazyConfig
from ssd import utils
np.random.seed(0)
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.functional as F
import requests
import torchvision
from PIL import Image

import functools
import time
import click
import pprint
import tops
import tqdm
from pathlib import Path
from ssd.evaluate import evaluate
from ssd import utils
from tops import logger, checkpointer
from torch.optim.lr_scheduler import ChainedScheduler
from omegaconf import OmegaConf

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image, preprocess_image

torch.backends.cudnn.benchmark = True

from ssd.modeling.retinanetOutputWrapper import RetinaNetOutputWrapper

from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget, ClassifierOutputTarget

from tops.misc import (
    get_config, get_trained_model, get_dataloader
)

def predict(input_tensor, model, detection_threshold, class_names):
    outputs = model(input_tensor)
    pred_classes = [class_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

def get_sample_data(cfg, dataset_to_visualize = "val"):

    cfg.train.batch_size = 1
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        if cfg.data_train.dataset._target_ == torch.utils.data.ConcatDataset:
            for dataset in cfg.data_train.dataset.datasets:
                dataset.transform.transforms = dataset.transform.transforms[:-1]
        else:
            cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        dataset = instantiate(cfg.data_train.dataloader)
        gpu_transform = instantiate(cfg.data_train.gpu_transform)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        dataset = instantiate(cfg.data_val.dataloader)
        gpu_transform = instantiate(cfg.data_val.gpu_transform)

    # Assumes that the first GPU transform is Normalize
    # If it fails, just change the index from 0.
    image_mean = torch.tensor(cfg.data_train.gpu_transform.transforms[0].mean).view(1, 3, 1, 1)
    image_std = torch.tensor(cfg.data_train.gpu_transform.transforms[0].std).view(1, 3, 1, 1)
    sample = next(iter(dataset))
    # print("sample:", sample)
    sample = gpu_transform(sample)

    print("The first sample in the dataset has the following keys:", sample.keys())
    for key, item in sample.items():
        print(
            key, ": shape=", item.shape if hasattr(item, "shape") else "",
            "dtype=", item.dtype if hasattr(item, "dtype") else type(item), sep="")

    return (sample, image_mean, image_std)


def draw_image(cfg, sample, image_mean, image_std):
    image = (sample["image"] * image_std + image_mean)
    image = (image*255).byte()[0]
    boxes = sample["boxes"][0]
    boxes[:, [0, 2]] *= image.shape[-1]
    boxes[:, [1, 3]] *= image.shape[-2]
    im = image.permute(1, 2, 0).cpu().numpy()
    im = draw_boxes(im, boxes.cpu().numpy(), sample["labels"][0].cpu().numpy().tolist(), class_name_map=cfg.label_map)

    plt.rc('figure', dpi=300)
    plt.imshow(im)
    plt.show()

def retinanet_reshape_transform(x):
    quality_lvl = 2 # lower level is higher quality
    target_size = x[quality_lvl].size()
    target_size = target_size[-2 : ]
    # print("target_size: ", target_size)

    activations = []
    for value in x:
    # for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations

def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """

    # labels = [str(label) for label in labels]
    # print("labels: ", labels)
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    # print("renormalized_cam.shape:", renormalized_cam.shape)
    # print("grayscale_cam:", grayscale_cam)
    images = []
    # print("renormalize_cam_in_bounding_boxes boxes:", type(boxes))
    boxes_int = np.rint(boxes)
    for box in boxes_int:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # print("renormalize_cam_in_bounding_boxes x1, y1, x2, y2:", x1, y1, x2, y2)
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

    renormalized_cam = np.max(np.float32(images), axis = 0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    # print("renormalized_cam:", renormalized_cam, "renormalized_cam.shape:", renormalized_cam.shape)

    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    # print("class_name_map:", class_name_map, type(class_name_map))

    return eigencam_image_renormalized

    # image_with_bounding_boxes = draw_boxes(eigencam_image_renormalized, boxes, labels, class_name_map=class_name_map)
    # image_with_bounding_boxes  = draw_boxes(cam_image, samble_boxes, sample_labels, class_name_map=cfg.label_map)


def get_cam_image(model, input_tensor, labels, boxes, norm = True, renorm=False):
    """
        Computes the CAM for a given image.

        Args:
            model: The model to use.
            input_tensor: The input tensor.
            labels: The labels of the image.
            boxes: The bounding boxes of the image.
            renorm: Whether to renormalize the CAM.
    """
    wrapped_model = RetinaNetOutputWrapper(model=model)
    wrapped_model = wrapped_model.eval().to(tops.get_device())

    # print(wrapped_model.feature_extractor)
    target_layers = [wrapped_model.feature_extractor]

    ############################# get activations #############################
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    cam = EigenCAM(model,
               target_layers,
               use_cuda=torch.cuda.is_available(),
               reshape_transform=retinanet_reshape_transform)
    cam.uses_gradients=False

    grayscale_cam = cam(input_tensor, targets=targets)
    # print("grayscale_cam:", grayscale_cam.shape)
    grayscale_cam = grayscale_cam[0, :]
    if norm:
        grayscale_cam = grayscale_cam / np.max(grayscale_cam)
    # print("grayscale_cam:", grayscale_cam.shape)

    # print("image.shape:", input_tensor.shape, "image type:", type(input_tensor))
    image = input_tensor[0]
    image_float_np = image.permute(1, 2, 0).cpu().numpy()

    if not renorm:
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    else:
        cam_image = renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam)

    return cam_image


@click.command()
@click.argument("config_path", default="configs/tdt4265_fpn.py", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def visualize_model_cam(config_path: Path):

    cfg = utils.load_config(config_path)
    print("cfg:", cfg.keys())

    ############################# get model #####################################

    tops.init(cfg.output_dir)
    tops.set_AMP(cfg.train.amp)
    tops.set_seed(cfg.train.seed)

    model = get_trained_model(cfg)

    ############################# get sample data #############################
    sample, image_mean, image_std = get_sample_data(cfg)

    input_tensor = (sample["image"] * image_std + image_mean)

    ############################# get preds #############################

    class_names = list(cfg["label_map"].values())
    print("class_names:", class_names)

    image = input_tensor[0]
    image_float_np = image.permute(1, 2, 0).cpu().numpy()
    print(image_float_np.shape)

    sample_boxes = sample["boxes"][0]
    sample_boxes[:, [0, 2]] *= image.shape[-1]
    sample_boxes[:, [1, 3]] *= image.shape[-2]
    sample_boxes = sample_boxes.cpu().numpy()

    sample_labels = sample["labels"][0].cpu().numpy().tolist()

    sample_classes = [class_names[i] for i in sample_labels]
    # print("samble_boxes:", type(sample_boxes), sample_boxes.shape)
    # print("sample_labels:", type(sample_labels), sample_labels)
    # print("sample_classes:", type(sample_classes), sample_classes)

    ################### draw image with bounding boxes #########################

    # get activations
    cam_image = get_cam_image(model, input_tensor, sample_labels, sample_boxes, renorm=False)

    plt.subplot(2, 1, 1)
    image_with_bounding_boxes  = draw_boxes(cam_image, sample_boxes, sample_labels, class_name_map=cfg.label_map)
    plt.title("Default CAM")
    plt.imshow(image_with_bounding_boxes)

    # get activations
    renorm = get_cam_image(model, input_tensor, sample_labels, sample_boxes, renorm=True)

    image_with_bounding_boxes  = draw_boxes(renorm, sample_boxes, sample_labels, class_name_map=cfg.label_map)
    plt.subplot(2, 1, 2)
    plt.title("Renormalized CAM")
    plt.imshow(image_with_bounding_boxes)
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    visualize_model_cam()
