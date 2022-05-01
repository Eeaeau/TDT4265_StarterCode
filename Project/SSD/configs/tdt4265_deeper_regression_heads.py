# from ssd.modeling import backbones
from .tdt4265_focal_loss import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    # model,
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
from ssd.modeling.retinanet import RetinaNet
from ssd.modeling import AnchorBoxes

# anchors.aspect_ratios = [ [2, 3] , [2, 3], [2, 3], [2, 3], [2, 3], [2, 3] ]
# anchors.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
aspect_ratios=[[3], [3], [3], [3], [2], [2]]
# anchors.feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]]

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,  # Add 1 for background
    anchor_prob_initialization = False
)
