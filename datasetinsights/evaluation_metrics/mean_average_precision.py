r"""Reference.

https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision\
Update algorithm from:
https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
"""

import numpy as np

from .average_precision_2d_bbox import AveragePrecisionBBox2D
from .base import EvaluationMetric


class MeanAveragePrecision(EvaluationMetric):
    """2D Bounding Box Mean Average Precision metrics.

    Implementation of classic mAP metrics. We use 10 IoU thresholds
    of .50:.05:.95. This is the same metrics in cocoEval.summarize():
    Average Precision (AP) @[IoU=0.50:0.95 | area=   all | maxDets=100]

    Attributes:
        label_records (dict): save prediction records for each label
        ap_method (string): AP interoperation method name for AP calculation
        {"EveryPointInterpolation"| "NPointInterpolatedAP"}
        gt_bboxes_count (dict): ground truth box count for each label
        iou_thresholds (numpy.array): iou thresholds

    Args:
        iou_start (float): iou range starting point (default: 0.5)
        iou_end (float): iou range ending point (default: 0.95)
        iou_step (float): iou step size (default: 0.05)
        interpolation (string): AP interoperation method name for AP calculation
    """

    def __init__(
        self,
        iou_start=0.5,
        iou_end=0.95,
        iou_step=0.05,
        interpolation="EveryPointInterpolation",
    ):
        self.interpolation = interpolation
        self.iou_thresholds = np.linspace(
            iou_start,
            iou_end,
            np.round((iou_end - iou_start) / iou_step) + 1,
            endpoint=True,
        )
        self.map_records = [
            AveragePrecisionBBox2D(iou, interpolation)
            for iou in self.iou_thresholds
        ]

    def reset(self):
        """Reset metrics."""
        self.map_records = [
            AveragePrecisionBBox2D(iou, self.interpolation)
            for iou in self.iou_thresholds
        ]

    def update(self, mini_batch):
        """Update records per mini batch

        Args:
            mini_batch (list(list)): a list which contains batch_size of
            gt bboxes and pred bboxes pair in each image.
            For example, if batch size = 2, mini_batch looks like:
            [[gt_bboxes1, pred_bboxes1], [gt_bboxes2, pred_bboxes2]]
            where gt_bboxes1, pred_bboxes1 contain gt bboxes and pred bboxes
            in one image
        """
        for mean_ap in self.map_records:
            mean_ap.update(mini_batch)

    def compute(self):
        """Compute AP for each label.

        Return:
            mAP (float): mean average precision across all ious
        """
        mean_sum = 0
        for mean_ap in self.map_records:
            mean_sum += np.mean(
                [
                    result_per_label
                    for result_per_label in mean_ap.compute().values()
                ]
            )
        return mean_sum / len(self.map_records)
