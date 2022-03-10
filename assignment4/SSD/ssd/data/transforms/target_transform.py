import torch
from ssd.modeling.anchor_encoder import AnchorEncoder



class GroundTruthBoxesToAnchors(torch.nn.Module):
    """
        
    """
    def __init__(self, anchors, iou_threshold: float):
        super().__init__()
        self.iou_threshold = iou_threshold

        self.anchors = anchors
        self.encoder = AnchorEncoder(self.anchors)

    @property
    def dboxes(self):
        return self.anchors

    def __call__(self, sample):
        bbox, label = self.encoder.encode(sample["boxes"], sample["labels"], self.iou_threshold)
        return dict(image=sample["image"], boxes=bbox, labels=label)
