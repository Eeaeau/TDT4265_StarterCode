import imp
from .tdt4265_fpn import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

from tops.config import LazyCall as L
from ssd.modeling import FocalLoss
from ssd.modeling.retinanetOutputWrapper import RetinaNetOutputWrapper
import torch

# print(model)

x = torch.randn(1, 3, 128, 1024)
wrappermodel = RetinaNetOutputWrapper(model)
print(wrappermodel.keys())

# wrappermodel(x)
