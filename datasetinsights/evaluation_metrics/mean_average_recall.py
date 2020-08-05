r"""Reference.

http://cocodataset.org/#detection-eval
https://arxiv.org/pdf/1502.05082.pdf
https://github.com/rafaelpadilla/Object-Detection-Metrics/issues/22
"""

import numpy as np

from .average_recall_2d_bbox import AverageRecallBBox2D
from .base import EvaluationMetric


class MeanAverageRecall(EvaluationMetric):
    """2D Bounding Box Mean Average Recall metrics.

    Implementation of classic mAR metrics. We use 10 IoU thresholds
    of .50:.05:.95. This is the same metrics in cocoEval.summarize():
    Average Recall (AR) @[IoU=0.50:0.95 | area = all | maxDets=100]

    Attributes:
        mar_records (dict): save prediction records for each label
        iou_thresholds (numpy.array): iou thresholds
        max_detections (int): max detections per image

    Args:
        iou_start (float): iou range starting point (default: 0.5)
        iou_end (float): iou range ending point (default: 0.95)
        iou_step (float): iou step size (default: 0.05)
        max_detections (int): max detections per image (default: 100)
    """

    def __init__(
        self, iou_start=0.5, iou_end=0.95, iou_step=0.05, max_detections=100
    ):
        self.iou_thresholds = np.linspace(
            iou_start,
            iou_end,
            np.round((iou_end - iou_start) / iou_step) + 1,
            endpoint=True,
        )
        self.max_detections = max_detections
        self.mar_records = [
            AverageRecallBBox2D(iou, max_detections)
            for iou in self.iou_thresholds
        ]

    def reset(self):
        """Reset metrics."""
        self.mar_records = [
            AverageRecallBBox2D(iou, self.max_detections)
            for iou in self.iou_thresholds
        ]

    def update(self, mini_batch):
        """Update records per mini batch.

        Args:
            mini_batch (list(list)): a list which contains batch_size of
            gt bboxes and pred bboxes pair in each image.
            For example, if batch size = 2, mini_batch looks like:
            [[gt_bboxes1, pred_bboxes1], [gt_bboxes2, pred_bboxes2]]
            where gt_bboxes1, pred_bboxes1 contain gt bboxes and pred bboxes
            in one image
        """
        for mean_ar in self.mar_records:
            mean_ar.update(mini_batch)

    def compute(self):
        """Compute AR for each label.

        Return:
            average_recall (dict): a dictionary of AR scores per label.
        """
        mean_sum = 0
        for mean_ar in self.mar_records:
            mean_sum += np.mean(
                [
                    result_per_label
                    for result_per_label in mean_ar.compute().values()
                ]
            )
        return mean_sum / len(self.mar_records)
