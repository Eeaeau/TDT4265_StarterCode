# Inherit configs from the default ssd300
import torchvision
import torch
from ssd.data import VOCDataset
from ssd.modeling import backbones
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)
# from .ssd300 import train, anchors, optimizer, schedulers, model, data_train, data_val, loss_objective
from .utils import get_dataset_dir


from .tdt4265_augmented import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    # backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

# Keep the model, except change the backbone and number of classes
model.feature_extractor = L(backbones.ResNet101)()

backbone
