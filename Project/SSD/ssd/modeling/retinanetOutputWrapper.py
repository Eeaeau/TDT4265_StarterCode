from queue import PriorityQueue
import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms

class RetinaNetOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feature_extractor = model.feature_extractor
        # print(self.model)
        # print("######################## wrapper #####################################")

    def forward(self, x):
        boxes, labels, scores = self.model(x)[0]
        # print("test")
        # print(x.shape)
        out_dict = dict({
            "boxes": boxes,
            "labels": labels,
            "scores": scores
        })
        # print(out_dict)
        return [out_dict]
