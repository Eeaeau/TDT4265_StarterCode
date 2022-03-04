# Modified from: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD/
import torch
import tops
import torch.nn.functional as F
from ssd import utils
from typing import Optional

def calc_iou_tensor(box1_ltrb, box2_ltrb):
    """ Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-src
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """

    N = box1_ltrb.size(0)
    M = box2_ltrb.size(0)

    be1 = box1_ltrb.unsqueeze(1).expand(-1, M, -1)
    be2 = box2_ltrb.unsqueeze(0).expand(N, -1, -1)

    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
    rb = torch.min(be1[:,:,2:], be2[:,:,2:])

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0]*delta[:,:,1]

    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]

    iou = intersect/(area1 + area2 - intersect)
    return iou

# This function is from https://github.com/kuangliu/pytorch-ssd.
class AnchorEncoder(object):
    """
        Transfer between (bboxes, labels) <-> SSD Output
    """

    def __init__(self, anchors):
        self.anchors = anchors(order="ltrb")
        self.anchors_xywh = tops.to_cuda(anchors(order="xywh").unsqueeze(dim=0))
        self.nboxes = self.anchors.size(0)
        self.scale_xy = anchors.scale_xy
        self.scale_wh = anchors.scale_wh

    def encode(self, bboxes_in: torch.Tensor, labels_in: torch.Tensor, iou_threshold: float):
        """
            Encode ground truth boxes and targets to anchors.
            Each ground truth is assigned to at least 1 anchor and
            each anchor is assigned to every ground truth if IoU threshold is met.
            
            Args:
                bboxes_in (num_targets, 4): ground truth boxes.
                labels_in (num_targets): labels of targets.
                iou_criteria: IoU threshold required to match a GT box to anchor
            Returns:
                boxes (num_priors, 4): real values for priors.
                labels (num_priros): labels for priors.
        """
        ious = calc_iou_tensor(bboxes_in, self.anchors)
        #ious: shape [batch_size, num_anchors]
        best_target_per_anchor, best_target_per_anchor_idx = ious.max(dim=0)
        best_anchor_per_target, best_anchor_per_target_idx = ious.max(dim=1)

        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_anchor.index_fill_(0, best_anchor_per_target_idx, 2.0)

        idx = torch.arange(0, best_anchor_per_target_idx.size(0), dtype=torch.int64)
        best_target_per_anchor_idx[best_anchor_per_target_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_target_per_anchor > iou_threshold
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_target_per_anchor_idx[masks]]
        bboxes_out = self.anchors.clone()
        bboxes_out[masks, :] = bboxes_in[best_target_per_anchor_idx[masks], :]
        # Transform format to xywh format
        bboxes_out = utils.bbox_ltrb_to_center(bboxes_out)
        return bboxes_out, labels_out

    def decode_output(self, bbox_delta: torch.Tensor, confs_in: Optional[torch.Tensor]):
        """
            Decodes SSD bbox delta/confidences to ltrb boxes.
            bbox_delta: [batch_size, 4, num_anchors], in center form (xywh)
            confs_in: [batch_size, num_classes, num_anchors]
        """
        bbox_delta = bbox_delta.permute(0, 2, 1)

        bbox_delta[:, :, :2] = self.scale_xy*bbox_delta[:, :, :2]
        bbox_delta[:, :, 2:] = self.scale_wh*bbox_delta[:, :, 2:]

        bbox_delta[:, :, :2] = bbox_delta[:, :, :2]*self.anchors_xywh[:, :, 2:] + self.anchors_xywh[:, :, :2]
        bbox_delta[:, :, 2:] = bbox_delta[:, :, 2:].exp()*self.anchors_xywh[:, :, 2:]

        boxes_ltrb = utils.bbox_center_to_ltrb(bbox_delta)
        if confs_in is not None:
            confs_in = confs_in.permute(0, 2, 1)
            confs_in = F.softmax(confs_in, dim=-1)
        return boxes_ltrb, confs_in