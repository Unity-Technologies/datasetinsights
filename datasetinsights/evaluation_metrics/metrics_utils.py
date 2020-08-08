import collections

import numpy as np


def mean_metrics_over_iou(metric_per_iou):
    """Calculate mean value over IOUs.

    Args:
        metric_per_iou (dict): metric records for each iou

    Returns (float):
        mean metric values over IOUs
    """
    mean_sum = 0
    for mean_ap in metric_per_iou:
        result = mean_ap.compute()
        mean_sum += np.mean(
            [result_per_label for result_per_label in result.values()]
        )
    return mean_sum / len(metric_per_iou)


def filter_pred_bboxes(pred_bboxes, max_detections):
    """Save bboxes with same label in to a dictionary.

    This operation only apply to predictions for a single image.

    Args:
        pred_bboxes (list): a list of prediction bounding boxes list
        max_detections (int): max detections per label

    Returns:
        labels (dict): a dictionary of prediction boundign boxes
    """
    labels = collections.defaultdict(list)
    for box in pred_bboxes:
        labels[box.label].append(box)
    for label, boxes in labels.items():
        boxes = sorted(boxes, key=lambda bbox: bbox.score, reverse=True)
        # only consider the top confident predictions
        if len(boxes) > max_detections:
            labels[label] = boxes[:max_detections]
        else:
            labels[label] = boxes

    return labels
