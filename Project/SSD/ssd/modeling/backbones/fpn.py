from queue import PriorityQueue
import torch
import torchvision.models as models
from torchvision.models.resnet import BasicBlock
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from typing import Tuple, List
from torch import nn

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# MaskRCNN requires a backbone with an attached FPN
class ResnetWithFPN(torch.nn.Module):
    def __init__(self, inp=torch.randn(1, 3, 128, 1024), model_version="resnet34", pretrained=True):
        super().__init__()
        # super(Resnet101WithFPN, self).__init__()

        #these two are changed on during the runs. the 2.3.1-2.3.2 does not use 1024 as the two last layers but uses 64
        self.out_channels = [256, 256, 256, 512, 1024, 1024]
        #self.out_channels = [256, 256, 256, 512, 64, 64]


        self.name = model_version +"WithFPN"
        print(f'Model used: {model_version}')
        m = torch.hub.load('pytorch/vision:v0.11.2', model_version, pretrained=pretrained) #'pytorch/vision:v0.11.2' used here.
        # if model_version == "resnet34":
        #     m = models.resnet34(pretrained=True)
        # elif model_version == "resnet50":
        #     m = models.resnet50(pretrained=True)
        # elif model_version == "resnet101":
        #     m = models.resnet101(pretrained=True)
        # elif model_version == "resnet152":
        #     m = models.resnet152(pretrained=True)
        # else:
        #     raise NotImplementedError("Only resnet50, resnet101, and resnet152 are supported")






        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})

        with torch.no_grad():
            out = self.body(inp)
            # print(out)
            # x = out["3"] #hva er poenget med denne?
            x = list(out.values())[-1]

        in_channels_list = [o.shape[1] for o in out.values()] #skal denne være inni with torch.no_grad?
        self.extras = nn.ModuleList([
            torch.nn.Sequential(
                BasicBlock (inplanes = in_channels_list[-1], planes = self.out_channels[-2], stride = 2,
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels_list[-1], out_channels=self.out_channels[-2], kernel_size=1, stride=2),
                    # nn.ReLU())),
                    nn.BatchNorm2d(self.out_channels[-2]),)),
                )
            ,
            torch.nn.Sequential(
                BasicBlock (inplanes = self.out_channels[-2], planes = self.out_channels[-1], stride = 2,
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels=self.out_channels[-2], out_channels=self.out_channels[-1], kernel_size=1, stride=2),
                    # nn.ReLU())),
                    nn.BatchNorm2d(self.out_channels[-1]),)),
                )
        ])

        # print(self.extras[0](x))

        with torch.no_grad():
            for i, extra in enumerate(self.extras):
                #print(i)
                x = extra(x)
                #print(x.shape)
                in_channels_list.append(x.shape[1])


        print("---------------------------------------", in_channels_list, "---------------------------------------")

        # Build FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=256)

        #self.out_channels = [64,128, 256, 512, 1024, 2048]
        self.out_channels = [256] * 6
        # self.out_channels = [256, 256, 256, 256, 64, 64]
        print("############################################################")
    def forward(self, x):
        # print("x: ", x.shape)
        # exit()
        features = []
        x = self.body(x)
        #print(x["3"].shape)

        #print("\n finishe body \n")

        for i, extra in enumerate(self.extras):

            x[f"{i+4}"] = extra(x[f"{i+3}"])

            #features.append(x)
        #print("\n finishe extras \n")

        x = self.fpn(x)
        # for i in range(6):
        #    print("fpn output: ", x[f"{i}"].shape)

        features.extend(x.values() )
        out_features = tuple(features)
        # self.output_feature_shape = [o.shape[-2:] for o in out_features]

        # for idx, feature in enumerate(features):
        #     out_channel = self.out_channels[idx]
        #     print("out_channel: ", out_channel, "\n")
        #     print("feature: ", feature, "\n")
        #     h, w = self.output_feature_shape[idx]
        #     expected_shape = (out_channel, h, w)
        #     assert feature.items().shape[1:] == expected_shape, \
        #         f"Expected shape: {expected_shape}, got: {feature.items().shape[1:]} at output IDX: {idx}"
        # assert len(out_features) == len(self.output_feature_shape),\
        #     f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        # exit()
        #print(out_features)

        # print(tuple(x.values())[0].shape)
        # exit()
        return tuple(x.values())

    # def reshape_transform(x, model):

    #     # target_size = x['feat3'].size()[-2 : ]
    #     target_layers = [model.layer4[-1]]
    #     input_tensor = # Create an input tensor image for your model..
    #     # Note: input_tensor can be a batch tensor with several images!

    #     cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    #     grayscale_cam = grayscale_cam[0, :]
    #     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


    #     activations = []
    #     for key, value in x.items():
    #         activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    #     activations = torch.cat(activations, axis=1)

    #     return activations

# Now we can build our model!
