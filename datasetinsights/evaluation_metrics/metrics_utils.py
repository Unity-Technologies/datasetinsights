import numpy as np


def mean_metrics_over_iou(metric_per_iou):
    """Calculate mean value over ious.

    Args:
        metric_per_iou (dict): metric records for each iou

    Returns:
        mean metric values over ious
    """
    mean_sum = 0
    for mean_ap in metric_per_iou:
        result = mean_ap.compute()
        mean_sum += np.mean(
            [result_per_label for result_per_label in result.values()]
        )
    return mean_sum / len(metric_per_iou)
