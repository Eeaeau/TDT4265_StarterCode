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
#from ssd.modeling.retinanetv2 import RetinaNet
from ssd.modeling import AnchorBoxes

<<<<<<< HEAD
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
=======
# anchors.aspect_ratios = [ [2, 3] , [2, 3], [2, 3], [2, 3], [2, 3], [2, 3] ]
# anchors.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
aspect_ratios=[[3], [3], [3], [3], [2], [2]]
# anchors.feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]]
>>>>>>> 2e8e0827b1775112f5a2510cb04ce2975160b76d

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,  # Add 1 for background
    anchor_prob_initialization = False
)