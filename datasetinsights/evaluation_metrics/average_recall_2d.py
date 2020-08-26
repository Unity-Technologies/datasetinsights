r"""Reference.

http://cocodataset.org/#detection-eval
https://arxiv.org/pdf/1502.05082.pdf
https://github.com/rafaelpadilla/Object-Detection-Metrics/issues/22
"""
import collections

import numpy as np

from datasetinsights.io.bbox import group_bbox2d_per_label

from .base import EvaluationMetric
from .records import Records


class AverageRecall(EvaluationMetric):
    """2D Bounding Box Average Recall metrics.

    This metric would calculate average recall (AR) for each label under
    an iou threshold (default: 0.5). The maximum number of detections
    per image is limited (default: 100).

    Args:
        iou_threshold (float): iou threshold (default: 0.5)
        max_detections (int): max detections per image (default: 100)
    """

    TYPE = "metric_per_label"

    def __init__(self, iou_threshold=0.5, max_detections=100):
        self._label_records = collections.defaultdict(
            lambda: Records(iou_threshold=self._iou_threshold)
        )
        self._gt_bboxes_count = collections.defaultdict(int)
        self._iou_threshold = iou_threshold
        self._max_detections = max_detections

    def reset(self):
        """Reset AR metrics."""
        self._label_records = collections.defaultdict(
            lambda: Records(iou_threshold=self._iou_threshold)
        )
        self._gt_bboxes_count = collections.defaultdict(int)

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
                self._gt_bboxes_count[gt_bbox.label] += 1

            pred_bboxes = sorted(
                pred_bboxes, key=lambda bbox: bbox.score, reverse=True
            )
            if len(pred_bboxes) > self._max_detections:
                pred_bboxes = pred_bboxes[: self._max_detections]
            bboxes_per_label = group_bbox2d_per_label(pred_bboxes)
            for label in bboxes_per_label:
                self._label_records[label].add_records(
                    gt_bboxes, bboxes_per_label[label]
                )

    def compute(self):
        """Compute AR for each label.

        Returns:
            dict: a dictionary of AR scores per label.
        """
        average_recall = {}
        _label_records = self._label_records
        for label in self._gt_bboxes_count:
            # if there are no predicted boxes with this label
            if label not in _label_records:
                average_recall[label] = 0
                continue

            match_results = _label_records[label].match_results
            _gt_bboxes_count = self._gt_bboxes_count[label]

            # The number of TP
            sum_tp = sum(list(zip(*match_results))[1])

            max_recall = sum_tp / _gt_bboxes_count

            average_recall[label] = max_recall

        return average_recall


class MeanAverageRecallAverageOverIOU(EvaluationMetric):
    """2D Bounding Box Mean Average Recall metrics.

    This implementation computes Mean Average Recall (mAR) metric,
    which is implemented as the Average Recall average over all
    labels and IOU = 0.5:0.95:0.05. The max detections
    per image is limited to 100.

    .. math:: mAR^{IoU=0.5:0.95:0.05} = mean_{label,IoU}
    AR^{label, IoU=0.5:0.95:0.05}
    """

    TYPE = "scalar"

    IOU_THRESHOULDS = np.linspace(
        0.5, 0.95, np.round((0.95 - 0.5) / 0.05) + 1, endpoint=True
    )

    def __init__(self):
        self._mar_per_iou = [
            AverageRecall(iou)
            for iou in MeanAverageRecallAverageOverIOU.IOU_THRESHOULDS
        ]

    def reset(self):
        [mean_ar.reset() for mean_ar in self._mar_per_iou]

    def update(self, mini_batch):
        for mean_ar in self._mar_per_iou:
            mean_ar.update(mini_batch)

    def compute(self):
        """Compute mAR over IOU.
        """
        result = np.mean(
            [
                value
                for ar in self._mar_per_iou
                for value in ar.compute().values()
            ]
        )
        return result
