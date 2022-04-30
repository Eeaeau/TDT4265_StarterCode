
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from torch.autograd import Variable

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)

def focal_loss(y, p, alpha=1, gamma=2):
    print(type(p))
    print(p.shape)
    print(f'p: {p}')
    print(type(y))
    print(y.shape)
    print(f'y: {y}')

    #weights = torch.pow(-)
    t = y.float() * (1 - p) ** gamma * p.log()
    return -t.sum()

class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors, gamma=2, eps=1e-7):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh
        self.alpha = [[[0.01], [1],[1],[1],[1],[1],[1],[1],[1]]]
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
        with torch.no_grad():
            hot_encoded = nn.functional.one_hot(gt_labels, num_classes=confs.shape[1]).transpose(1,2)
            to_log = - F.log_softmax(confs, dim=1)
            p_k = - F.softmax(confs, dim=1)
            #mask = hard_negative_mining(to_log, gt_labels, 3.0)
            alpha = torch.as_tensor([[[0.01], [1],[1],[1],[1],[1],[1],[1],[1]]]).to(p_k.device)
            weight = torch.pow(1.0-p_k, self.gamma)
            focal = -alpha * weight * to_log

            #loss_tmp = torch.einsum('bc...,bc...->b...', (hot_encoded, focal))
            loss_tmp = torch.sum(hot_encoded * focal, dim=1)
            loss = torch.mean(loss_tmp)
            #mask = focal_loss(to_log, gt_labels, alpha=self.alpha)
        #classification_loss = F.cross_entropy(confs, gt_labels, reduction="none")
        #classification_loss = classification_loss[mask].sum()
        print(f'loss: {loss}')
        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]

        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]

        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        #total_loss = regression_loss/num_pos + classification_loss/num_pos
        total_loss = regression_loss/num_pos + loss/num_pos
        classification_loss=loss/num_pos

        to_log = dict(
            regression_loss=regression_loss/num_pos,
            #classification_loss=classification_loss/num_pos,
            classification_loss=loss/num_pos,

            total_loss=total_loss
        )

        print(f'Classification loss: {classification_loss}')
        print(f'Total loss: {total_loss}')

        return total_loss, to_log
