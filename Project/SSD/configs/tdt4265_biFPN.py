from .tdt4265_init_weights import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    #backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

from tops.config import LazyCall as L
from ssd.modeling.backbones import BiFPN

anchors.aspect_ratios = [ [2, 3] , [2, 3], [2, 3], [2, 3], [2, 3], [2, 3] ]

backbone = L(BiFPN)()
#backbone = L(BiFPNv2)()