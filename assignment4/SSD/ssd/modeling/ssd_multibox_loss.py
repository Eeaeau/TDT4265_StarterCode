import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


class SSDMultiboxLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)


    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]
        with torch.no_grad():
            to_log = - F.log_softmax(confs, dim=1)[:, 0]
            mask = hard_negative_mining(to_log, gt_labels, 3.0)
        classification_loss = F.cross_entropy(confs, gt_labels, reduction="none")
        classification_loss = classification_loss[mask].sum()

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        total_loss = regression_loss/num_pos + classification_loss/num_pos
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=classification_loss/num_pos,
            total_loss=total_loss
        )
        return total_loss, to_log
