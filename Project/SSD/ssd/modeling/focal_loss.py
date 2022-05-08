import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from torch.autograd import Variable


def focal_loss(confs, gt_labels, alpha, gamma=2):


    hot_encoded = F.one_hot(gt_labels, num_classes=confs.shape[1]).transpose(1,2)

    log_pk = F.log_softmax(confs, dim=1)
    p_k = F.softmax(confs, dim=1)
    alpha = torch.tensor(alpha).reshape((1, 9, 1)).to(p_k.device)
    #alpha = torch.tensor([[10] + 8*[1000]]).reshape((1, 9, 1)).to(p_k.device)
    #alpha = torch.tensor([[[10],[1000],[1000],[1000],[1000],[1000],[1000],[1000],[1000]]]).to(p_k.device)
    weight = torch.pow(1.0-p_k, gamma)
    focal = -alpha * weight * log_pk
    #print(hot_encoded.shape)
    #print(focal.shape)
    loss_tmp = hot_encoded*focal
    focal_loss = loss_tmp.sum(dim=1).mean()

    return focal_loss



class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors, alpha, gamma=2, eps=1e-7):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh
        self.alpha=alpha
        #self.alpha = [[[0.01], [1],[1],[1],[1],[1],[1],[1],[1]]]
        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        self.gamma = gamma
        self.eps = eps

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
        #with torch.no_grad():

        loss = focal_loss(confs, gt_labels, alpha=self.alpha)

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]

        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]

        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        #total_loss = regression_loss/num_pos + classification_loss/num_pos
        total_loss = regression_loss/num_pos + loss
        classification_loss=loss

        to_log = dict(
            regression_loss=regression_loss/num_pos,
            #classification_loss=classification_loss/num_pos,
            classification_loss=loss,

            total_loss=total_loss
        )

        # print(f'Classification loss: {classification_loss}')
        # print(f'Total loss: {total_loss}')

        return total_loss, to_log
