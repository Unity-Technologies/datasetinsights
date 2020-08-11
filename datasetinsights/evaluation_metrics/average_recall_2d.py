r"""Reference.

http://cocodataset.org/#detection-eval
https://arxiv.org/pdf/1502.05082.pdf
https://github.com/rafaelpadilla/Object-Detection-Metrics/issues/22
"""
import collections

import numpy as np

from .base import EvaluationMetric
from .metrics_utils import filter_pred_bboxes, mean_metrics_over_iou
from .records import Records


class AverageRecall(EvaluationMetric):
    """2D Bounding Box Average Recall metrics.

    Attributes:
        label_records (dict): save prediction records for each label
        gt_bboxes_count (dict): ground truth box count for each label
        iou_threshold (float): iou threshold
        max_detections (int): max detections per image

    Args:
        iou_threshold (float): iou threshold (default: 0.5)
        max_detections (int): max detections per image (default: 100)
    """

    COMPUTE_TYPE = "metric_per_label"

    def __init__(self, iou_threshold=0.5, max_detections=100):
        self.label_records = collections.defaultdict(
            lambda: Records(iou_threshold=self.iou_threshold)
        )
        self.gt_bboxes_count = collections.defaultdict(int)
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def reset(self):
        """Reset AR metrics."""
        self.label_records = collections.defaultdict(
            lambda: Records(iou_threshold=self.iou_threshold)
        )
        self.gt_bboxes_count = collections.defaultdict(int)

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
        for bboxes in mini_batch:
            gt_bboxes, pred_bboxes = bboxes
            for gt_bbox in gt_bboxes:
                self.gt_bboxes_count[gt_bbox.label] += 1

            bboxes_per_label = filter_pred_bboxes(
                pred_bboxes, self.max_detections
            )
            for label in bboxes_per_label:
                self.label_records[label].add_records(
                    gt_bboxes, bboxes_per_label[label]
                )

    def compute(self):
        """Compute AR for each label.

        Return:
            average_recall (dict): a dictionary of AR scores per label.
        """
        average_recall = {}
        label_records = self.label_records
        for label in self.gt_bboxes_count:
            # if there are no predicted boxes with this label
            if label not in label_records:
                average_recall[label] = 0
                continue

            pred_infos = label_records[label].pred_infos
            gt_bboxes_count = self.gt_bboxes_count[label]

            # The number of TP
            sum_tp = sum(list(zip(*pred_infos))[1])

            max_recall = sum_tp / gt_bboxes_count

            average_recall[label] = max_recall

        return average_recall


class MeanAverageRecallAverageOverIOU(EvaluationMetric):
    """2D Bounding Box Mean Average Recall metrics.

    This implementation computes Mean Average Recall (mAR) metric,
    which is implemented as the Average Recall average over all
    labels and IOU thresholds [0.5:0.95:0.05]. The max detections
    per image is limited to 100.
    mAR = mean_{label, IOU}AR(label, IOU)

    Attributes:
        mar_per_iou (dict): save prediction records for each label

    Args:
        iou_start (float): iou range starting point (default: 0.5)
        iou_end (float): iou range ending point (default: 0.95)
        iou_step (float): iou step size (default: 0.05)
        max_detections (int): max detections per image (default: 100)
    """

    COMPUTE_TYPE = "float"

    IOU_THRESHOULDS = np.linspace(
        0.5, 0.95, np.round((0.95 - 0.5) / 0.05) + 1, endpoint=True
    )

    def __init__(self):
        self.mar_per_iou = [
            AverageRecall(iou)
            for iou in MeanAverageRecallAverageOverIOU.IOU_THRESHOULDS
        ]

    def reset(self):
        """Reset metrics."""
        [mean_ar.reset() for mean_ar in self.mar_per_iou]

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
        for mean_ar in self.mar_per_iou:
            mean_ar.update(mini_batch)

    def compute(self):
        """Compute AR for each label.

        Returns (float):
            mean average recall over ious
        """
        return mean_metrics_over_iou(self.mar_per_iou)
