from ssd.modeling import backbones
from .tdt4265_fpn import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    data_train,
    backbone,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)


from tops.config import LazyCall as L
from ssd.modeling.backbones import Resnet101WithFPN


loss_objective = L(FocalLoss)
# backbone = L(Resnet101WithFPN)()
# backbone = L(Resnet101WithFPN)(
#     image_channels=3,
#     model_type='resnet101',
#     pretrained=True)
