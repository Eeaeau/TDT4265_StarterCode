from ssd.modeling import backbones
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
import torchvision

from tops.config import LazyCall as L
from ssd.modeling.backbones import Resnet101WithFPN

# backbone = L(Resnet101WithFPN)()
backbone = L(Resnet101WithFPN)(
    image_channels=3,
    output_feature_sizes="${anchors.feature_sizes}",
    model_type='resnet101',
    pretrained=True)
