import numpy as np

from .average_precision_2d import AveragePrecision
from .base import EvaluationMetric


class PrecisionRecallRecords(EvaluationMetric):
    """Precision and recall records for each class

    This implementation would record precision and recall information for
    each label.
    """

    TYPE = "precision_recall"

    def __init__(self, iou_threshold=0.5):
        self._ap = AveragePrecision(iou_threshold=iou_threshold)

    def reset(self):
        self._ap.reset()

    def update(self, mini_batch):
        self._ap.update(mini_batch)

    def compute(self):
        """Compute precision and recall information for each label

        Returns:
            dict: a dict of {label_id: (precision, recall)} mapping
        """
        pr_results = {}
        _label_records = self._ap._label_records

        for label in self._ap._gt_bboxes_count:
            # if there are no predicted boxes with this label
            if label not in _label_records:
                pr_results[label] = [0, 0]
                continue
            match_results = _label_records[label].match_results
            _gt_bboxes_count = self._ap._gt_bboxes_count[label]

            match_results = sorted(match_results, reverse=True)
            true_pos = np.array(list(zip(*match_results))[1]).astype(int)
            false_pos = 1 - true_pos

            acc_tp = np.cumsum(true_pos)
            acc_fp = np.cumsum(false_pos)

            recall = acc_tp / _gt_bboxes_count
            precision = np.divide(acc_tp, (acc_fp + acc_tp))
            pr_results[label] = (precision, recall)
        return pr_results
