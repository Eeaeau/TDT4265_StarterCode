import torch
import torch.nn as nn
from typing import Tuple, List

#As the layers are pretty much the same except for the first layer I think it is nicer to create a class that contains the backbbones of the layers, I have seen other networks created like this(ex mobilenet)
class InitLayer(torch.nn.Sequential):
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
            nn.Hardswish(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=max_pool_stride),
            
            # nn.Conv2d(in_channels=32, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.Hardswish(inplace=True),
            # nn.BatchNorm2d(128),
            #nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=max_pool_stride),

            # nn.Conv2d(in_channels=128, out_channels=512, kernel_size=kernel_size, padding=padding),
            # nn.Hardswish(inplace=True),
            # nn.Dropout(0.2),
            
            #nn.BatchNorm2d(512),
            #nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=1, padding=1),
            # nn.Conv2d(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.Hardswish(inplace=True),
            # nn.Dropout(0.2),
            
            #nn.BatchNorm2d(256),
            # #nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=1, padding=1),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.ReLU(),
            # #nn.Dropout(0.2),
            # nn.BatchNorm2d(256),

            #nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=max_pool_stride),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=max_pool_stride),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Hardswish(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=num_out_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.Hardswish(inplace=True),
        )

#dont quite understand why its a relu output of the init layer and as an input and output in the rest of the layers, removed it as i got an error about size
class ConvLayer(torch.nn.Sequential):
    def __init__(self,
            num_in_channels,
            num_out_channels,
            kernel_size=3,
            stride1=1,
            stride2=2,
            padding1=1,
            padding2=1
            ):
        super().__init__(
            nn.Conv2d(in_channels=num_in_channels, out_channels=num_in_channels, kernel_size=kernel_size, stride=stride1, padding=padding1),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels=num_in_channels, out_channels=num_out_channels, kernel_size=kernel_size, stride=stride2, padding=padding2),
            nn.SiLU(),
        )

class ModelV1(torch.nn.Module):
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
        self.model.append(InitLayer(image_channels, output_channels[0]))
        self.model.append(ConvLayer(output_channels[0], output_channels[1]))
        self.model.append(ConvLayer(output_channels[1], output_channels[2]))
        self.model.append(ConvLayer(output_channels[2], output_channels[3]))
        self.model.append(ConvLayer(output_channels[3], output_channels[4]))
        self.model.append(ConvLayer(output_channels[4], output_channels[5], stride2=1, padding2=0))


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
        out_features = []
        for layer in self.model:
            #print(layer)
            x = layer(x)
            out_features.append(x)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)
