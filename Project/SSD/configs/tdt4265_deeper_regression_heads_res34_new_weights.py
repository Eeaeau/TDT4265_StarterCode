# from ssd.modeling import backbones
from .tdt4265_focal_loss_res34_new_weights import (
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
#from ssd.modeling.retinanetv2 import RetinaNet
from ssd.modeling import AnchorBoxes

anchors.aspect_ratios = [ [2, 3] , [2, 3], [2, 3], [2, 3], [2, 3], [2, 3] ]
# anchors.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
#anchors.aspect_ratios=[[2, 2], [2, 3], [2, 3], [2, 3], [2, 2], [2, 2]]
#anchors.aspect_ratios=[[2], [3], [3], [3], [2], [2]]
# model = L(RetinaNet)(
#     feature_extractor="${backbone}",
#     anchors="${anchors}",
#     loss_objective="${loss_objective}",
#     num_classes=8 + 1,  # Add 1 for background
#     anchor_prob_initialization = False
# )

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,  # Add 1 for background
    anchor_prob_initialization = False
)