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

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM, EigenCAM

torch.backends.cudnn.benchmark = True

from ssd.modeling.retinanetOutputWrapper import RetinaNetOutputWrapper

config_path = "configs/tdt4265.py"
# cfg = LazyConfig.load(config_path)
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget, ClassifierOutputTarget

from performance_assessment.save_comparison_images import get_config, get_trained_model, get_dataloader

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

def get_sample_data(cfg, dataset_to_visualize = "train"):


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

    # print("dataset:", dataset)
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
    print("x: ", x)
    print("x keys: ", x.keys())
    # target_size = x[2].size()[-2 : ]
    target_size = x["0"].size()[-2 : ]
    print(target_size)
    target_size = target_size[-2 : ]
    print(target_size)

    activations = []
    for value in x:
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))

def visualize_model_cam(config_path: Path):
    if config_path is None:
        config_path = "configs/tdt4265_fpn.py"
    # print("config_path:", config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cfg = get_config(config_path)
    # model = get_trained_model(cfg)
    cfg = utils.load_config(config_path)
    print("cfg:", cfg.keys())

    tops.init(cfg.output_dir)
    tops.set_AMP(cfg.train.amp)
    tops.set_seed(cfg.train.seed)

    # model = tops.to_cuda(instantiate(cfg.model))
    model = get_trained_model(cfg)
    # cfg = get_config(config_path)
    # model = get_trained_model(cfg)

    wrapped_model = RetinaNetOutputWrapper(model=model)
    wrapped_model = wrapped_model.model.eval().to(device)
    # print(wrapped_model)

    # Get your input
    # img = read_image("data/tdt4265_2022/images/train/trip007_glos_Video00000_3.png")
    # input_tensor = normalize(img.float()/255, mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225])

    # input_tensor = input_tensor.to(device)
    # # Add a batch dimension:
    # input_tensor = input_tensor.unsqueeze(0)

    # get sample data
    sample, image_mean, image_std = get_sample_data(cfg)

    input_tensor = (sample["image"] * image_std + image_mean)
    input_tensor = input_tensor.to(device)
    print("input_tensor shape:", input_tensor.shape)
    print("input_tensor:", input_tensor)


    # draw image
    # draw_image(cfg, sample, image_mean, image_std)
    # labels = sample["labels"][0].cpu().numpy().tolist()
    class_names = list(cfg["label_map"].values())
    print("label_map:", class_names)
    # This will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    boxes, classes, labels, indices = predict(input_tensor, wrapped_model, 0.1, class_names)

    target_layers = [wrapped_model.feature_extractor.fpn]

    print("target_layers:", target_layers)

    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    cam = EigenCAM(model,
               target_layers,
               use_cuda=torch.cuda.is_available(),
               reshape_transform=retinanet_reshape_transform)
    cam.uses_gradients=False

    grayscale_cam = cam(input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    print("grayscale_cam:", grayscale_cam.shape)

    # Take the first image in the batch:
    image = (sample["image"] * image_std + image_mean)[0]
    # image = (image*255).byte()[0]
    image_float_np = image.permute(1, 2, 0).cpu().numpy()
    # image_float_np = image.cpu().numpy().reshape( (128,1024,3) )
    # image_float_np = image.cpu().numpy()
    print(image_float_np.shape)
    # plt.imshow(image_float_np)
    # plt.show()

    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.show()
    # im = draw_boxes(im, boxes.cpu().numpy(), sample["labels"][0].cpu().numpy().tolist(), class_name_map=cfg.label_map)

    # image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    # Image.fromarray(image_with_bounding_boxes)


if __name__ == "__main__":
    np.random.seed(0)
    visualize_model_cam()
