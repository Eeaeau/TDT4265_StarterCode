import torch
import numpy as np
from typing import Union

def bbox_ltrb_to_ltwh(boxes_ltrb: Union[np.ndarray, torch.Tensor]):
    cat = torch.cat if isinstance(boxes_ltrb, torch.Tensor) else np.concatenate
    assert boxes_ltrb.shape[-1] == 4
    return cat((boxes_ltrb[..., :2], boxes_ltrb[..., 2:] - boxes_ltrb[..., :2]), -1)
    
def bbox_center_to_ltrb(boxes_center: Union[np.ndarray, torch.Tensor]):
    cat = torch.stack if isinstance(boxes_center, torch.Tensor) else np.stack
    assert boxes_center.shape[-1] == 4
    cx, cy, w, h = [boxes_center[..., i] for i in range(4)]
    return cat((
        cx - 0.5*w,
        cy - 0.5*h,
        cx + 0.5*w,
        cy + 0.5*h,
    ), -1)

def bbox_center_to_ltrb(boxes_center: Union[np.ndarray, torch.Tensor]):
    cat = torch.stack if isinstance(boxes_center, torch.Tensor) else np.stack
    assert boxes_center.shape[-1] == 4
    cx, cy, w, h = [boxes_center[..., i] for i in range(4)]
    return cat((
        cx - 0.5*w,
        cy - 0.5*h,
        cx + 0.5*w,
        cy + 0.5*h,
    ), -1)

def bbox_ltrb_to_center(boxes_lrtb: Union[np.ndarray, torch.Tensor]):
    cat = torch.stack if isinstance(boxes_lrtb, torch.Tensor) else np.stack
    assert boxes_lrtb.shape[-1] == 4
    l, t, r, b = [boxes_lrtb[..., i] for i in range(4)]
    return cat((
        0.5*(l+r),
        0.5*(t+b),
        r - l,
        b - t
    ), -1)