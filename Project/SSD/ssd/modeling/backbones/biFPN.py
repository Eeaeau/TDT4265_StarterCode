import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.resnet import BasicBlock
"""
This is a modieied version of the code provided in the following link:
https://github.dev/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py

We have only adapted the code to work with the rest of our model.

"""

class DepthwiseConv(nn.Module):
    """
    Depthwise separable convolution, as expleind on page 4 section 3.3 in the paper
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=1, stride=1, padding=0):
        super(DepthwiseConv,self).__init__()

        if out_channels == None:
            out_channels = in_channels

        self.depthConv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.pointConv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = self.depthConv(inputs)
        x = self.pointConv(x)
        x = self.batchnorm(x)
        return self.activation(x)

class BiFPNLayer(nn.Module):
    """
    Implementation of the BiFPN layers.
    """
    def __init__(self, feature_size=256, epsilon=0.0001):
        super(BiFPNLayer, self).__init__()
        self.epsilon = epsilon

        self.p3_td =  DepthwiseConv(feature_size)
        self.p4_td = DepthwiseConv(feature_size)
        self.p5_td = DepthwiseConv(feature_size)
        self.p6_td = DepthwiseConv(feature_size)
        self.p7_td = DepthwiseConv(feature_size)

        self.p4_out = DepthwiseConv(feature_size)
        self.p5_out = DepthwiseConv(feature_size)
        self.p6_out = DepthwiseConv(feature_size)
        self.p7_out = DepthwiseConv(feature_size)
        self.p8_out = DepthwiseConv(feature_size)

        self.w1_td = torch.Tensor(2, 5)
        nn.init.kaiming_uniform_(self.w1_td, nonlinearity='relu')#
        self.w1_relu = nn.ReLU()

        self.w2_up = torch.Tensor(3, 5)
        nn.init.kaiming_uniform_(self.w2_up, nonlinearity='relu')
        self.w2_relu = nn.ReLU()

    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x, p8_x = inputs
        #Fast normalized fusion
        w1_td = self.w1_relu(self.w1_td)
        w1_td /= torch.sum(w1_td, dim=0) + self.epsilon

        w2_up = self.w2_relu(self.w2_up)
        w2_up /= torch.sum(w2_up, dim=0) + self.epsilon


        #print(p7_x.shape)
        # Calculate Top-Down Pathway
        #The top-down pathway is calculated by multiplying the feature maps of the previous layers with the corresponding weight matrix.
        #The resulting feature maps are then concatenated and passed through a convolutional layer.
        #exleind on page 4 section 3.3 in the paper

        p8_td = p8_x
        p7_td = self.p7_td(w1_td[0, 0] * p7_x + w1_td[1, 0] * F.interpolate(p8_td, scale_factor=2))
        p6_td = self.p6_td(w1_td[0, 1] * p6_x + w1_td[1, 1] * F.interpolate(p7_td, scale_factor=2))
        p5_td = self.p5_td(w1_td[0, 2] * p5_x + w1_td[1, 2] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1_td[0, 3] * p4_x + w1_td[1, 3] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1_td[0, 4] * p3_x + w1_td[1, 4] * F.interpolate(p4_td, scale_factor=2))

        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2_up[0, 0] * p4_x + w2_up[1, 0] * p4_td + w2_up[2, 0] * nn.Upsample(scale_factor=0.5,recompute_scale_factor=True)(p3_out))
        p5_out = self.p5_out(w2_up[0, 1] * p5_x + w2_up[1, 1] * p5_td + w2_up[2, 1] * nn.Upsample(scale_factor=0.5,recompute_scale_factor=True)(p4_out))

        p6_out = self.p6_out(w2_up[0, 2] * p6_x + w2_up[1, 2] * p6_td + w2_up[2, 2] * nn.Upsample(scale_factor=0.5,recompute_scale_factor=True)(p5_out))
        p7_out = self.p7_out(w2_up[0, 3] * p7_x + w2_up[1, 3] * p7_td + w2_up[2, 3] * nn.Upsample(scale_factor=0.5,recompute_scale_factor=True)(p6_out))
        p8_out = self.p8_out(w2_up[0, 4] * p8_x + w2_up[1, 4] * p8_td + w2_up[2, 4] * nn.Upsample(scale_factor=0.5,recompute_scale_factor=True)(p7_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out]

class BiFPN(nn.Module):
    """
    Implementation of the BiFPN.
    With resnet backbone.
    """

    def __init__(self, pretrained: bool=True,

                fpn_out_channels: int = 256,
                model_version:str = "resnet34",):
        super(BiFPN, self).__init__()

        self.fpn_out_channels = fpn_out_channels
        self.out_channels = [self.fpn_out_channels]*6

        model_version = "resnet101"

        self.name = model_version +"WithFPN"

        m = torch.hub.load('pytorch/vision:v0.11.2', model_version, pretrained=pretrained) #'pytorch/vision:v0.11.2' used here.
        inp=torch.randn(1, 3, 128, 1024)

        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})

        with torch.no_grad():
            out = self.body(inp)

            x = list(out.values())[-1]

        in_channels_list = [o.shape[1] for o in out.values()]



        feature_size =256
        self.p3 = nn.Conv2d( in_channels_list[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d( in_channels_list[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d( in_channels_list[2], feature_size, kernel_size=1, stride=1, padding=0)

        self.p6 = nn.Conv2d( in_channels_list[3], feature_size, kernel_size=1, stride=1, padding=0)
        self.p7 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p7 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p8 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)




        self.extras = nn.ModuleList([
            torch.nn.Sequential(
                BasicBlock (inplanes =  in_channels_list[-1], planes = feature_size, stride = 2,
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels= in_channels_list[-1], out_channels=feature_size, kernel_size=1, stride=2),
                    # nn.ReLU())),
                    nn.BatchNorm2d(feature_size),)),
                )
            ,
            torch.nn.Sequential(
                BasicBlock (inplanes = feature_size, planes = feature_size, stride = 2,
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=1, stride=2),
                    # nn.ReLU())),
                    nn.BatchNorm2d(feature_size),)),
                )
        ])



        bifpns = []
        for _ in range(3):
            bifpns.append(BiFPNLayer(256))
        self.bifpn = nn.Sequential(*bifpns)

    def forward(self, x):

        features = []

        x = self.body(x)
        for i, extra in enumerate(self.extras):

            x[f"{i+4}"] = extra(x[f"{i+3}"])



        ## Calculate the input column of BiFPN
        p3_x = self.p3(x["0"])
        """
        Works
        """
        p4_x = self.p4(x["1"])

        p5_x = self.p5(x["2"])

        p6_x = self.p6(x["3"])

        #p7_x = self.p7(p6_x)
        p7_x = self.p7(x["4"])
        p8_x = self.p8(p7_x)








        features = [p3_x, p4_x, p5_x, p6_x, p7_x,p8_x]

        out_features = self.bifpn(features)

        return tuple(out_features)
