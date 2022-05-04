# Modified from: https://github.com/lufficc/SSD
import torch
from typing import List
from math import sqrt

# Note on center/size variance:
# This is used for endcoding/decoding the regressed coordinates from the SSD bounding box head to actual locations.
# It's a trick to improve gradients from bounding box regression. Take a look at this post about more info:
# https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
class AnchorBoxes(object):
    def __init__(self, 
            image_shape: tuple, 
            feature_sizes: List[tuple], 
            min_sizes: List[int],
            strides: List[tuple],
            aspect_ratios: List[int],
            scale_center_variance: float,
            scale_size_variance: float):
        """Generate SSD anchors Boxes.
            It returns the center, height and width of the anchors. The values are relative to the image size
            Args:
                image_shape: tuple of (image height, width)
                feature_sizes: each tuple in the list is the feature shape outputted by the backbone (H, W)
            Returns:
                anchors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        self.scale_center_variance = scale_center_variance
        self.scale_size_variance = scale_size_variance
        self.num_boxes_per_fmap = [2 + 2*len(ratio) for ratio in aspect_ratios]
        # Calculation method slightly different from paper

        anchors = []
        # size of feature and number of feature
        for fidx, [fH, fW] in enumerate(feature_sizes):
            bbox_sizes = []
            h_min = min_sizes[fidx][0] / image_shape[0]
            w_min = min_sizes[fidx][1] / image_shape[1]
            bbox_sizes.append((w_min, h_min))
            h_max = sqrt(min_sizes[fidx][0]*min_sizes[fidx+1][0]) / image_shape[0]
            w_max = sqrt(min_sizes[fidx][1]*min_sizes[fidx+1][1]) / image_shape[1]
            bbox_sizes.append((w_max, h_max))
            for r in aspect_ratios[fidx]:
                h = h_min*sqrt(r)
                w = w_min/sqrt(r)
                bbox_sizes.append((w_min/sqrt(r), h_min*sqrt(r)))
                bbox_sizes.append((w_min*sqrt(r), h_min/sqrt(r)))
            scale_y = image_shape[0] / strides[fidx][0]
            scale_x = image_shape[1] / strides[fidx][1]
            for w, h in bbox_sizes:
                for i in range(fH):
                    for j in range(fW):
                        cx = (j + 0.5)/scale_x
                        cy = (i + 0.5)/scale_y
                        anchors.append((cx, cy, w, h))

        self.anchors_xywh = torch.tensor(anchors).clamp(min=0, max=1).float()
        self.anchors_ltrb = self.anchors_xywh.clone()
        self.anchors_ltrb[:, 0] = self.anchors_xywh[:, 0] - 0.5 * self.anchors_xywh[:, 2]
        self.anchors_ltrb[:, 1] = self.anchors_xywh[:, 1] - 0.5 * self.anchors_xywh[:, 3]
        self.anchors_ltrb[:, 2] = self.anchors_xywh[:, 0] + 0.5 * self.anchors_xywh[:, 2]
        self.anchors_ltrb[:, 3] = self.anchors_xywh[:, 1] + 0.5 * self.anchors_xywh[:, 3]

    def __call__(self, order):
        if order == "ltrb":
            return self.anchors_ltrb
        if order == "xywh":
            return self.anchors_xywh

    @property
    def scale_xy(self):
        return self.scale_center_variance

    @property
    def scale_wh(self):
        return self.scale_size_variance