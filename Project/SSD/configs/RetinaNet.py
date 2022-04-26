# Inherit configs from the default ssd300
import torchvision
import torch
from ssd.data import VOCDataset
from ssd.modeling import backbones
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)
from .ssd300 import train, anchors, optimizer, schedulers, model, data_train, data_val, loss_objective
from .utils import get_dataset_dir


model.feature_extractor = torchvision.ops.feature_pyramid_network(
    in_channels_list=(3, 300, 300),
    output_shape=(1, 256, 256),
    num_features=256,
    freeze_bn=True,
    num_classes=21,
)
