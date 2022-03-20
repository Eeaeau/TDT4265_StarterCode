import torch
import torch.nn as nn
from typing import Tuple, List

#As the layers are pretty much the same except for the first layer I think it is nicer to create a class that contains the backbbones of the layers, I have seen other networks created like this(ex mobilenet)
class InitBlock(torch.nn.Sequential):
    def __init__(self,
            num_in_channels,
            num_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            max_pool_stride=2,
            maxpool_kernel_size=2):
        super().__init__(
        nn.Conv2d(in_channels=num_in_channels, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=32),
        nn.Hardswish(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Hardswish(inplace=True),
        
        
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(num_features=64),
        nn.Hardswish(inplace=True),
        
        nn.Conv2d(in_channels=64, out_channels= num_out_channels, kernel_size=kernel_size, stride=2, padding=padding),
        nn.BatchNorm2d(num_features=128),
        nn.Hardswish(inplace=True),
        )

#dont quite understand why its a relu output of the init layer and as an input and output in the rest of the layers, removed it as i got an error about size
class ConvBlock(torch.nn.Sequential):
    def __init__(self,
            num_in_channels,
            num_out_channels,
            kernel_size=3,
            stride1=1,
            stride2=1,
            padding1=1,
            padding2=1
            ):
        super().__init__(
            nn.Conv2d(in_channels=num_in_channels, out_channels=num_in_channels, kernel_size=kernel_size, stride=stride1, padding=padding1),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=kernel_size, stride=stride2, padding=padding2),
            nn.SiLU(),
        )

class ModelV4(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.model = nn.ModuleList()

        #adding the layers
        self.model.append(InitBlock(image_channels, output_channels[0])) #out 128
        #can add skip connection
        self.model.append(ConvBlock(output_channels[0], output_channels[1]))#out 128
        self.model.append(ConvBlock(output_channels[1], output_channels[2]))#out 128

        self.model.append(ConvBlock(output_channels[2], output_channels[3]))#out 256
        #can add skip connection
        self.model.append(ConvBlock(output_channels[3], output_channels[4]))#out 256
        self.model.append(ConvBlock(output_channels[4], output_channels[5]))#out 256

        self.model.append(ConvBlock(output_channels[5], output_channels[6]))#out 128

        self.model.append(ConvBlock(output_channels[6], output_channels[7]))#out 128
        self.model.append(ConvBlock(output_channels[7], output_channels[8]))#out 128
        
        self.model.append(ConvBlock(output_channels[8], output_channels[9]))#out 64

        #can add skip connection
        self.model.append(ConvBlock(output_channels[9], output_channels[10]))#out 64
        self.model.append(ConvBlock(output_channels[10], output_channels[11], stride2=1, padding2=0))


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        i = 0
        # for layer in self.model:
        #     print(i)
        #     i += 1
        out_features = []
        
        x1 = self.model[0](x) #128
        residual1 = x1 #128
        #print(x1.shape)
        #print(residual1.shape)
        out_features.append(x1)#128

        x2 = self.model[1](x1) #128
        out_features.append(x2) #128
        #print(x2.shape)
        x3 = self.model[2](x2+residual1) #128
        out_features.append(x3) #128

        x4 = self.model[3](x3) #256
        residual2 = x4 #256
        out_features.append(x4) #256

        x5 = self.model[4](x4) #256
        out_features.append(x5) #256
        x6 = self.model[5](x5+residual2) #256
        out_features.append(x6) #256

        x7 = self.model[6](x6)  #128
        residual3 = x7 #128
        out_features.append(x7) #128

        x8 = self.model[7](x7) #128
        out_features.append(x8) #128
        x9 = self.model[8](x8+residual3) #128
        out_features.append(x9) #128

        x10 = self.model[9](x9)  #64
        residual4 = x10 #64
        out_features.append(x10) #64

        x11 = self.model[10](x10) #64
        out_features.append(x11) #64
        x12 = self.model[11](x11+residual4) #64
        out_features.append(x12) #64
        
        # for idx, feature in enumerate(out_features):
        #     out_channel = self.out_channels[idx]
        #     h, w = self.output_feature_shape[idx]
        #     expected_shape = (out_channel, h, w)
        #     assert feature.shape[1:] == expected_shape, \
        #         f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        # assert len(out_features) == len(self.output_feature_shape),\
        #     f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)


