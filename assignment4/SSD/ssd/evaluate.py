import torch
import tqdm
import numpy as np
import tops
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from ssd import utils
from tops import logger
import sys
import os


def silent_evaluation(eval_object):
    """Runs coco evaluation without writing to screen
    """
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")

    eval_object.evaluate()
    eval_object.accumulate()
    eval_object.summarize()

    sys.stdout = old_stdout # reset old stdout


def calculate_class_aps(coco_gt, coco_dt, label_map):
    out_stats = {}

    print("---------------------------------------------------")
    for index, class_name in label_map.items():
        eval_object = COCOeval(coco_gt, coco_dt, iouType='bbox')
        eval_object.params.catIds = [index]
        silent_evaluation(eval_object)

        ap_score = eval_object.stats[0]  # We should consider changing this to MaP@iou=0.5

        extra_message = ""
        if ap_score == -1:
            extra_message = "(No objects of this class in validation set)"

        if ap_score != -1:
            stat_key = f"AP_{class_name}"
            out_stats[stat_key] = ap_score

        print("AP for class", class_name, "is", f"{eval_object.stats[0]:.4f}", extra_message)

    return out_stats


@torch.no_grad()
def evaluate(
        model,
        dataloader: torch.utils.data.DataLoader,
        cocoGt: COCO,
        gpu_transform: torch.nn.Module,
        label_map):
    """
        Evaluates over dataloader and returns COCO stats
    """
    model.eval()
    ret = []
    for batch in tqdm.tqdm(dataloader, desc="Evaluating on dataset"):
        batch["image"] = tops.to_cuda(batch["image"])
        batch = gpu_transform(batch)
        with torch.cuda.amp.autocast(enabled=tops.AMP()):
            predictions = model(batch["image"], nms_iou_threshold=0.50, max_output=200,
                score_threshold=0.05)

        for idx in range(len(predictions)):
            boxes_ltrb, categories, scores = predictions[idx]
            # ease-of-use for specific predictions
            H, W = batch["height"][idx], batch["width"][idx]
            box_ltwh = utils.bbox_ltrb_to_ltwh(boxes_ltrb)
            box_ltwh[:, [0, 2]] *= W
            box_ltwh[:, [1, 3]] *= H
            box_ltwh, category, score = [x.cpu() for x in [box_ltwh, categories, scores]]
            img_id = batch["image_id"][idx].item()
            for b_ltwh, label_, prob_ in zip(box_ltwh, category, score):
                ret.append([img_id, *b_ltwh.tolist(), prob_.item(),
                            int(label_)])
    model.train()
    final_results = np.array(ret).astype(np.float32)
    if final_results.shape[0] == 0:
        logger.log("WARNING! There were no predictions with score > 0.05. This indicates a bug in your code.")
        return dict()
    cocoDt = cocoGt.loadRes(final_results)
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 16 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]
    E.evaluate()
    E.accumulate()
    E.summarize()

    class_ap_stats = calculate_class_aps(cocoGt, cocoDt, label_map)

    stats_all_objects = {
        "mAP": E.stats[0], # same as mAP@
        "mAP@0.5": E.stats[1], # Same as PASCAL VOC mAP
        "mAP@0.75": E.stats[2],
        "mAP_small": E.stats[3],
        "mAP_medium": E.stats[4],
        "mAP_large": E.stats[5],
        "average_recall@1": E.stats[6],
        "average_recall@10": E.stats[7],
        "average_recall@100": E.stats[8],
        "average_recall@100_small": E.stats[9],
        "average_recall@100_medium": E.stats[10],
        "average_recall@100_large": E.stats[11],
    }

    return dict(stats_all_objects, **class_ap_stats)
