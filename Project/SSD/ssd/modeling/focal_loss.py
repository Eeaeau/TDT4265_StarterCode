
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
        # self.alpha = [[[0.01], [1],[1],[1],[1],[1],[1],[1],[1]]]
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
        # confs = confs.transpose(1, 2) # [batch_size, num_anchors, num_classes]

        print_calculation = False

        # with torch.no_grad():
        # batch_size = gt_labels[0]
        # print("batch_size:", batch_size)
        # hot_encoded = nn.functional.one_hot(gt_labels, num_classes=confs.shape[1]).reshape(confs.shape)
        hot_encoded = nn.functional.one_hot(gt_labels, num_classes=confs.shape[1]).transpose(1, 2)

        sm_output = F.softmax(confs, dim=1)

        # to_log = - F.log_softmax(confs, dim=1)[:, 0]
        # to_log = F.log_softmax(confs, dim=1)

        #mask = hard_negative_mining(to_log, gt_labels, 3.0)

        # alpha = torch.tensor([[[0.01], [1],[1],[1],[1],[1],[1],[1],[1]]]).to(p_k.device)
        # alpha = torch.tensor([[10] + 8*[1000]]).reshape((1, 9, 1)).to(sm_output.device)
        alpha = torch.as_tensor([[[10],[1000],[1000],[1000],[1000],[1000],[1000],[1000],[1000]]]).to(sm_output.device)


        focal = torch.pow(1.0-sm_output, self.gamma)

        log_sm_output = torch.log(sm_output)

        focal_loss = - hot_encoded * alpha * focal * torch.log(sm_output)

        if print_calculation:
            print("hot_encoded shape:", hot_encoded.shape)
            print("hot_encoded:", hot_encoded[0])

            print("sm_output shape: ", sm_output.shape)
            print("sm_output: ", sm_output[0])

            print("alpha shape: ", alpha.shape)
            print("alpha: ", alpha)

            print("focal shape: ", focal.shape)
            print("focal: ", focal[0])

            print("log_sm_output shape: ", log_sm_output.shape)
            print("log_sm_output: ", log_sm_output[0])
            print("focal_loss shape: ", focal_loss.shape)
            print("focal_loss: ", focal_loss[0])

        classification_loss = focal_loss.sum(dim=1).mean()

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

        print(f'Total loss: {total_loss}')
        return total_loss, to_log


# import torch.nn as nn
# import torch
# import math
# import torch.nn.functional as F

# from torch.autograd import Variable

# def one_hot(index, classes):
#     size = index.size() + (classes,)
#     view = index.size() + (1,)

#     mask = torch.Tensor(*size).fill_(0)
#     index = index.view(*view)
#     ones = 1.

#     if isinstance(index, Variable):
#         ones = Variable(torch.Tensor(index.size()).fill_(1))
#         mask = Variable(mask, volatile=index.volatile)

#     return mask.scatter_(1, index, ones)

# def focal_loss(confs, gt_labels, alpha, gamma=2):

#     hot_encoded = nn.functional.one_hot(gt_labels, num_classes=confs.shape[1])
#     #torch.set_printoptions(profile="full")
#     #print(gt_labels[1])
#     #print(hot_encoded[1])

#     confs=confs.transpose(2,1)
#     #print(confs.shape)
#     #print(hot_encoded.shape)
#     log_pk=F.log_softmax(confs, dim=1)
#     p_k = F.softmax(confs, dim=1)
#     alpha = torch.as_tensor([0.01,1,1,1,1,1,1,1,1]).to(p_k.device)
#     #print(alpha.shape)
#     weight = torch.pow(1.0-p_k, gamma)
#     focal = -alpha * weight*hot_encoded* log_pk

#     loss = -torch.sum(focal,dim=1)

#     focal_loss = loss.mean()
#     #print(focal_loss)



#     return focal_loss

# def focal_loss_2(confs, gt_labels, alpha, gamma=2):
#     #torch.set_printoptions(profile="full")
#     #print(confs.shape[1])
#     hot_encoded = nn.functional.one_hot(gt_labels, num_classes=confs.shape[1]).transpose(1,2)

#     #confs=confs.transpose(2,1)
#     print(confs.shape)
#     print(hot_encoded.shape)
#     log_pk=F.log_softmax(confs, dim=1)
#     p_k = F.softmax(confs, dim=1)

#     alpha = torch.as_tensor([[[10],[1000],[1000],[1000],[1000],[1000],[1000],[1000],[1000]]]).to(p_k.device)

#     print(alpha.shape)
#     weight = torch.pow(1.0-p_k, gamma)
#     focal = -alpha * weight * log_pk
#     loss_tmp   =hot_encoded*focal
#     focal_loss = loss_tmp.sum(dim=1).mean()
#     #print(focal_loss)




#     return focal_loss



# class FocalLoss(nn.Module):
#     """
#         Implements the loss as the sum of the followings:
#         1. Confidence Loss: All labels, with hard negative mining
#         2. Localization Loss: Only on positive labels
#         Suppose input dboxes has the shape 8732x4
#     """
#     def __init__(self, anchors, gamma=2, eps=1e-7):
#         super().__init__()
#         self.scale_xy = 1.0/anchors.scale_xy
#         self.scale_wh = 1.0/anchors.scale_wh
#         self.alpha = [[[0.01], [1],[1],[1],[1],[1],[1],[1],[1]]]
#         self.sl1_loss = nn.SmoothL1Loss(reduction='none')
#         self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
#             requires_grad=False)
#         self.gamma = gamma
#         self.eps = eps

#     def _loc_vec(self, loc):
#         """
#             Generate Location Vectors
#         """
#         gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
#         gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
#         return torch.cat((gxy, gwh), dim=1).contiguous()

#     def forward(self,
#             bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
#             gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
#         """
#         NA is the number of anchor boxes (by default this is 8732)
#             bbox_delta: [batch_size, 4, num_anchors]
#             confs: [batch_size, num_classes, num_anchors]
#             gt_bbox: [batch_size, num_anchors, 4]
#             gt_label = [batch_size, num_anchors]
#         """
#         gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]
#         #with torch.no_grad():

#         loss = focal_loss_2(confs, gt_labels, alpha=self.alpha)


#         pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
#         bbox_delta = bbox_delta[pos_mask]

#         gt_locations = self._loc_vec(gt_bbox)
#         gt_locations = gt_locations[pos_mask]

#         regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
#         num_pos = gt_locations.shape[0]/4
#         #total_loss = regression_loss/num_pos + classification_loss/num_pos
#         total_loss = regression_loss/num_pos + loss
#         classification_loss=loss

#         to_log = dict(
#             regression_loss=regression_loss/num_pos,
#             #classification_loss=classification_loss/num_pos,
#             classification_loss=loss,

#             total_loss=total_loss
#         )

#         print(f'Classification loss: {classification_loss}')
#         print(f'Total loss: {total_loss}')

#         return total_loss, to_log
