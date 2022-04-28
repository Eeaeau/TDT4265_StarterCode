import torch.nn as nn
import torchvision
import torch
from typing import Tuple, List


# class FPN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.vgg = nn.ModuleList(list(torchvision.models.vgg16(pretrained=True).features)[:-1])
#         self.vgg[16].ceil_mode = True

class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.ModuleList(list(torchvision.models.resnet101(pretrained=True).features)[:-1])
        self.resnet[16].ceil_mode = True

        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3,stride=1,padding=1,dilation=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, kernel_size=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3,),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.ReLU(),
            ),
        ])
        # self.l2_norm = L2Norm(512, scale=20)
        self.init_parameters()
        self.out_channels = [512, 1024, 512, 256, 256, 256]

    def init_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self, state_dict):
        self.resnet.load_state_dict(state_dict)

    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)  # Conv4_3 L2 normalization
        features.append(s)


        for i in range(23, len(self.resnet)):
            x = self.resnet[i](x)
        for extra in self.extras:
            x = extra(x)
            features.append(x)

        return tuple(features)
