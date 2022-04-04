import torchvision
import torch
import numpy as np
import random

class ToTensor:
    def __call__(self, sample):
        sample["image"] = torch.from_numpy(np.rollaxis(sample["image"], 2, 0)).float() / 255
        if "boxes" in sample:
            sample["boxes"] = torch.from_numpy(sample["boxes"])
            sample["labels"] = torch.from_numpy(sample["labels"])
        return sample


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class RandomSampleCrop(torch.nn.Module):
    """Crop
    Implementation originally from: https://github.com/lufficc/SSD

    NOTE: This function needs to be run before to_tensor
    Arguments:
        sample dict containing at least the following:
        img (np.ndarray): the image being input during training
        boxes (np.ndarray): the original bounding boxes in pt form
        labels (np.ndarray): the class labels for each bbox
    Return:
        the same sample dict with modified img, boxes and labels
    """

    def __init__(self):
        super().__init__()
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, sample):
        image = sample["image"]
        boxes = sample["boxes"]
        labels = sample["labels"]
        # guard against no boxes
        if boxes is not None and boxes.shape[0] == 0:
            return sample
        height, width, _ = image.shape
        original_aspect_ratio = height / width

        boxes = boxes.copy()
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return sample

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < (original_aspect_ratio / 2) or h / w > (original_aspect_ratio * 2):
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                    rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                    rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                current_boxes[:, [0, 2]] /= w
                current_boxes[:, [1, 3]] /= h

                sample["image"] = current_image
                sample["boxes"] = current_boxes
                sample["labels"] = current_labels
                return sample


class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    def __call__(self, sample):
        image = sample["image"]
        if np.random.uniform() < self.p:
            sample["image"] = image.flip(-1)
            boxes = sample["boxes"]
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
            sample["boxes"] = boxes
        return sample


class Resize(torch.nn.Module):

    def __init__(self, imshape) -> None:
        super().__init__()
        self.imshape = tuple(imshape)

    @torch.no_grad()
    def forward(self, batch):
        batch["image"] = torchvision.transforms.functional.resize(batch["image"], self.imshape, antialias=True)
        return batch
