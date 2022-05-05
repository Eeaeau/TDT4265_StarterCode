import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms

class RetinaNetOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(RetinaNetOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]
