import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x1_t, y1_t, x2_t, y2_t = gt_box #sol
    x1_p, y1_p, x2_p, y2_p = prediction_box #sol
    if x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t:#sol
        return 0.0#sol

    # Compute intersection
    x1i = max(x1_t, x1_p)#sol
    x2i = min(x2_t, x2_p)#sol
    y1i = max(y1_t, y1_p)#sol
    y2i = min(y2_t, y2_p)#sol
    intersection = (x2i - x1i) * (y2i - y1i)#sol

    # Compute union
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)#sol
    gt_area = (x2_t - x1_t) * (y2_t - y1_t)#sol
    union = pred_area + gt_area - intersection#sol
    iou = 0
    iou = intersection / union#sol
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:#sol
        return 1#sol
    return num_tp / (num_fp + num_tp)#sol
    raise NotImplementedError


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    denominator = num_tp + num_fn#sol
    if denominator == 0:#sol
        return 0#sol
    return num_tp / denominator#sol
    raise NotImplementedError


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    ious = []#sol
    indices = []#sol
    # Find all possible matches with a IoU >= iou threshold
    for pred_idx, pred_box in enumerate(prediction_boxes):#sol
        for gt_idx, gt_box in enumerate(gt_boxes):#sol
            iou = calculate_iou(pred_box, gt_box)#sol
            if iou >= iou_threshold:#sol
                ious.append(iou)#sol
                indices.append((pred_idx, gt_idx))#sol

    ious = np.array(ious)#sol
    indices = np.array(indices)#sol
    if indices.size == 0:#sol
        return np.array([]), np.array([])#sol
    assert indices.shape[1] == 2#sol

    # Sort all matches on IoU in descending order
    sorted_idx = np.argsort(ious)[::-1]#sol
    ious = ious[sorted_idx]#sol
    indices = indices[sorted_idx]#sol

    # Find all matches with the highest IoU threshold
    seen_prediction_boxes = np.zeros(len(prediction_boxes))#sol
    seen_gt_boxes = np.zeros(len(gt_boxes))#sol
    final_prediction_boxes = []#sol
    final_gt_boxes = []#sol

    for (pred_idx, gt_idx) in indices:#sol
        if seen_prediction_boxes[pred_idx] == 0 and seen_gt_boxes[gt_idx] == 0:#sol
            final_prediction_boxes.append(prediction_boxes[pred_idx])#sol
            final_gt_boxes.append(gt_boxes[gt_idx])#sol

            seen_prediction_boxes[pred_idx] = 1#sol
            seen_gt_boxes[gt_idx] = 1#sol

    return np.array(final_prediction_boxes), np.array(final_gt_boxes)#sol
    return np.array([]), np.array([])


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    matched_pred_boxes, matched_gt_boxes = get_all_box_matches( #sol
        prediction_boxes, gt_boxes, iou_threshold)#sol

    num_tp = len(matched_gt_boxes)#sol
    num_fp = len(prediction_boxes) - num_tp#sol
    num_fn = len(gt_boxes) - num_tp#sol
    return {"true_pos": num_tp, "false_pos": num_fp, "false_neg": num_fn}#sol
    raise NotImplementedError


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    true_pos = 0#sol
    false_pos = 0#sol
    false_neg = 0#sol
    for idx in range(len(all_prediction_boxes)):#sol
        pbox = all_prediction_boxes[idx]#sol
        gtbox = all_gt_boxes[idx]#sol
        res = calculate_individual_image_result(pbox, gtbox, iou_threshold)#sol
        true_pos += res["true_pos"]#sol
        false_pos += res["false_pos"]#sol
        false_neg += res["false_neg"]#sol
    precision = calculate_precision(true_pos, false_pos, false_neg)#sol
    recall = calculate_recall(true_pos, false_pos, false_neg)#sol
    return precision, recall #sol
    raise NotImplementedError


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []
    for confidence_thr in confidence_thresholds:#sol
        temp_pred_box = []#sol
        for image_idx in range(len(all_prediction_boxes)):#sol
            pred_boxes = all_prediction_boxes[image_idx]#sol
            scores = confidence_scores[image_idx]#sol
            prune_mask = scores >= confidence_thr#sol
            temp_pred_box.append(pred_boxes[prune_mask])#sol
        precision, recall = calculate_precision_recall_all_images(#sol
            temp_pred_box, all_gt_boxes, iou_threshold#sol
        )#sol
        precisions.append(precision)#sol
        recalls.append(recall)#sol
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    final_precisions = []#sol
    for recall_level in recall_levels:#sol
        precision = precisions[recalls >= recall_level]#sol
        if precision.size == 0:#sol
            precision = 0#sol
        else:#sol
            precision = max(precision)#sol
        final_precisions.append(precision)#sol
    average_precision = 0
    average_precision = np.mean(final_precisions)#sol
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
