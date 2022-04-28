import torch.nn as nn
import torchvision
import torch
from typing import Tuple, List

# To assist you in designing the feature extractor you may want to print out
# the available nodes for resnet50.
m = torchvision.models.resnet101(pretrained=True)
train_nodes, eval_nodes = torchvision.models.feature_extraction.get_graph_node_names(m)

# To specify the nodes you want to extract, you could select the final node
# that appears in each of the main layers:
return_nodes = {
    # node_name: user-specified key for output dict
    'layer1.2.relu_2': 'layer1',
    'layer2.3.relu_2': 'layer2',
    'layer3.5.relu_2': 'layer3',
    'layer4.2.relu_2': 'layer4',
}

# But `create_feature_extractor` can also accept truncated node specifications
# like "layer1", as it will just pick the last node that's a descendent of
# of the specification. (Tip: be careful with this, especially when a layer
# has multiple outputs. It's not always guaranteed that the last operation
# performed is the one that corresponds to the output you desire. You should
# consult the source code for the input model to confirm.)
return_nodes = {
    'layer1': 'layer1',
    'layer2': 'layer2',
    'layer3': 'layer3',
    'layer4': 'layer4',
}

# Now you can build the feature extractor. This returns a module whose forward
# method returns a dictionary like:
# {
#     'layer1': output of layer 1,
#     'layer2': output of layer 2,
#     'layer3': output of layer 3,
#     'layer4': output of layer 4,
# }
# print(torchvision.models.feature_extraction.create_feature_extractor(m, return_nodes=return_nodes))

# Let's put all that together to wrap resnet101 with MaskRCNN

# MaskRCNN requires a backbone with an attached FPN
class Resnet101WithFPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # super(Resnet101WithFPN, self).__init__()
        # self.out_channels = output_channels
        # self.output_feature_shape = output_feature_sizes

        # Get a resnet101 backbone
        m = torchvision.models.resnet101(pretrained=True)
        # Extract 4 main layers
        self.body = torchvision.models.feature_extraction.create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        # self.resnet = nn.ModuleList(list(torchvision.models.resnet101(pretrained=True).features)[:-2])
        # print(self.resnet)
        # self.body = torchvision.models.feature_extraction.create_feature_extractor(m)
        # Dry run to get number of channels for FPN
        inp = torch.randn(1, 3, 128, 1024)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        print(in_channels_list)
        # Build FPN
        # self.out_channels = [128, 256, 128, 128, 64, 64]
        self.out_channels = 4
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels)

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        print(x)
        out_features = [x]

        # for i in range(len(x)):
        #     x= self.fpn(x)[str(i)]

        # out_features.append(x)

        # for idx, feature in enumerate(out_features):
        #     out_channel = self.out_channels[idx]
        #     h, w = self.output_feature_shape[idx]
        #     expected_shape = (out_channel, h, w)
        #     assert feature.shape[1:] == expected_shape, \
        #         f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        # assert len(out_features) == len(self.output_feature_shape),\
        #     f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

# # Now we can build our model!
# model = torchvision.models.detection.mask_rcnn.MaskRCNN(Resnet101WithFPN(), num_classes=8+1).eval()

# print(model(torch.randn(1, 3, 128, 1024)))
