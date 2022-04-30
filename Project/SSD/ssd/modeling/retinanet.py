
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
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

def focal_loss(confs, gt_labels, alpha, gamma=2):

    hot_encoded = nn.functional.one_hot(gt_labels, num_classes=confs.shape[1])
    #torch.set_printoptions(profile="full")
    #print(gt_labels[1])
    #print(hot_encoded[1])

    confs=confs.transpose(2,1)
    #print(confs.shape)
    #print(hot_encoded.shape)
    log_pk=F.log_softmax(confs, dim=1)
    p_k = F.softmax(confs, dim=1)
    alpha = torch.as_tensor([0.01,1,1,1,1,1,1,1,1]).to(p_k.device)
    #print(alpha.shape)
    weight = torch.pow(1.0-p_k, gamma)
    focal = -alpha * weight*hot_encoded* log_pk
    
    loss = -torch.sum(focal,dim=1)
    
    focal_loss = loss.mean()
    #print(focal_loss)
    

    
    return focal_loss

def focal_loss_2(confs, gt_labels, alpha, gamma=2):
    #torch.set_printoptions(profile="full")
    #print(confs.shape[1])
    hot_encoded = nn.functional.one_hot(gt_labels, num_classes=confs.shape[1]).transpose(1,2)
    
    #confs=confs.transpose(2,1)
    #print(confs.shape)
    #print(hot_encoded.shape)
    log_pk=F.log_softmax(confs, dim=1)
    p_k = F.softmax(confs, dim=1)

    alpha = torch.as_tensor(alpha).to(p_k.device)
    
    #rint(alpha.shape)
    weight = torch.pow(1.0-p_k, gamma)
    focal = -alpha * weight * log_pk
    loss_tmp   =hot_encoded*focal
    focal_loss = loss_tmp.sum(dim=1).mean()
    #print(focal_loss)
    
    

    
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
        self.alpha = alpha
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
           
        loss = focal_loss_2(confs, gt_labels, alpha=self.alpha)
        
        
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

        #print(f'Classification loss: {classification_loss}')
        #print(f'Total loss: {total_loss}')

        return total_loss, to_log