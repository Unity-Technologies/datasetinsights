r"""Reference.

https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision\
Update algorithm from:
https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
"""
import collections

import numpy as np

from datasetinsights.io.bbox import group_bbox2d_per_label

from .base import EvaluationMetric
from .records import Records


class AveragePrecision(EvaluationMetric):
    """2D Bounding Box Average Precision metrics.

    This metric would calculate average precision (AP) for each label under
    an iou threshold (default: 0.5). The maximum number of detections
    per image is limited (default: 100).

    Args:
        iou_threshold (float): iou threshold (default: 0.5)
        interpolation (string): AP interoperation method name for AP calculation
        max_detections (int): max detections per image (default: 100)
    """

    TYPE = "metric_per_label"

    def __init__(
        self,
        iou_threshold=0.5,
        interpolation="EveryPointInterpolation",
        max_detections=100,
    ):
        if interpolation == "EveryPointInterpolation":
            self._ap_method = self.every_point_interpolated_ap
        elif interpolation == "NPointInterpolatedAP":
            self._ap_method = self.n_point_interpolated_ap
        else:
            raise ValueError(f"Unknown AP method name: {interpolation}!")

        self._iou_threshold = iou_threshold
        self._max_detections = max_detections
        self._label_records = collections.defaultdict(
            lambda: Records(iou_threshold=self._iou_threshold)
        )
        self._gt_bboxes_count = collections.defaultdict(int)

    def reset(self):
        """Reset AP metrics."""
        self._label_records = collections.defaultdict(
            lambda: Records(iou_threshold=self._iou_threshold)
        )
        self._gt_bboxes_count = collections.defaultdict(int)

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
        for bboxes in mini_batch:
            gt_bboxes, pred_bboxes = bboxes

            pred_bboxes = sorted(
                pred_bboxes, key=lambda bbox: bbox.score, reverse=True
            )
            if len(pred_bboxes) > self._max_detections:
                pred_bboxes = pred_bboxes[: self._max_detections]

            bboxes_per_label = group_bbox2d_per_label(pred_bboxes)
            for label, boxes in bboxes_per_label.items():
                self._label_records[label].add_records(gt_bboxes, boxes)

            for gt_bbox in gt_bboxes:
                self._gt_bboxes_count[gt_bbox.label] += 1

    def compute(self):
        """Compute AP for each label.

        Returns:
            dict: a dictionary of AP scores per label.
        """
        average_precision = {}
        _label_records = self._label_records

        for label in self._gt_bboxes_count:
            # if there are no predicted boxes with this label
            if label not in _label_records:
                average_precision[label] = 0
                continue
            match_results = _label_records[label].match_results
            _gt_bboxes_count = self._gt_bboxes_count[label]

            match_results = sorted(match_results, reverse=True)
            true_pos = np.array(list(zip(*match_results))[1]).astype(int)
            false_pos = 1 - true_pos

            acc_tp = np.cumsum(true_pos)
            acc_fp = np.cumsum(false_pos)

            recall = acc_tp / _gt_bboxes_count
            precision = np.divide(acc_tp, (acc_fp + acc_tp))
            ap = self._ap_method(recall, precision)
            # add class result in the dictionary to be returned
            average_precision[label] = ap

        return average_precision

    @staticmethod
    def every_point_interpolated_ap(recall, precision):
        """Calculating the interpolation performed in all points.

        Args:
            recall (list): recall history of the prediction
            precision (list): precision history of the prediction

        Returns:
            float: average precision for all points interpolation
        """
        # TODO: make it readable
        mrec = [0] + list(recall) + [1]
        mpre = [0] + list(precision) + [0]
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

        return ap

    @staticmethod
    def n_point_interpolated_ap(recall, precision, point=11):
        """Calculating the n-point interpolation.

        Args:
            recall (list): recall history of the prediction
            precision (list): precision history of the prediction
            point (int): n, n-point interpolation

        Returns:
            float: average precision for n-point interpolation
        """
        # TODO: make it readable
        mrec = [e for e in recall]
        mpre = [e for e in precision]
        recall_values = np.linspace(0, 1, point)
        recall_values = list(recall_values[::-1])
        rho_interp = []
        recall_valid = []
        # For each recall_values (0, 0.1, 0.2, ... , 1)
        for r in recall_values:
            # Obtain all recall values higher or equal than r
            arg_greater_recalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if arg_greater_recalls.size != 0:
                pmax = max(mpre[arg_greater_recalls.min() :])
            recall_valid.append(r)
            rho_interp.append(pmax)
        ap = sum(rho_interp) / point
        # Generating values for the plot
        rvals = []
        rvals.append(recall_valid[0])
        [rvals.append(e) for e in recall_valid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rho_interp]
        pvals.append(0)

        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)

        return ap


class AveragePrecisionIOU50(EvaluationMetric):
    """2D Bounding Box Average Precision at IOU = 50%.

    This implementation would calculate AP at IOU = 50% for each label.
    """

    TYPE = "metric_per_label"

    def __init__(self):
        self._ap = AveragePrecision(iou_threshold=0.5)

    def reset(self):
        self._ap.reset()

    def update(self, mini_batch):
        self._ap.update(mini_batch)

    def compute(self):
        return self._ap.compute()


class MeanAveragePrecisionIOU50(EvaluationMetric):
    """2D Bounding Box Mean Average Precision metrics at IOU=50%.

    This implementation would calculate mAP at IOU=50%.

    .. math:: mAP^{IoU=50} = mean_{label}AP^{label, IoU=50}
    """

    TYPE = "scalar"

    def __init__(self):
        self._ap = AveragePrecision(iou_threshold=0.5)

    def reset(self):
        self._ap.reset()

    def update(self, mini_batch):
        self._ap.update(mini_batch)

    def compute(self):
        result = self._ap.compute()
        mean_ap = np.mean(
            [result_per_label for result_per_label in result.values()]
        )
        return mean_ap


class MeanAveragePrecisionAverageOverIOU(EvaluationMetric):
    """2D Bounding Box Mean Average Precision metrics.

    This implementation computes Mean Average Precision (mAP) metric,
    which is implemented as the Average Precision average over all
    labels and IOU = 0.5:0.95:0.05. The max detections per image is
    limited to 100.

    .. math:: mAP^{IoU=0.5:0.95:0.05} = mean_{label,IoU}
    AP^{label, IoU=0.5:0.95:0.05}
    """

    TYPE = "scalar"

    IOU_THRESHOULDS = np.linspace(
        0.5, 0.95, np.round((0.95 - 0.5) / 0.05) + 1, endpoint=True
    )

    def __init__(self):
        self._map_per_iou = [
            AveragePrecision(iou) for iou in self.IOU_THRESHOULDS
        ]

    def reset(self):
        [mean_ap.reset() for mean_ap in self._map_per_iou]

    def update(self, mini_batch):
        for mean_ap in self._map_per_iou:
            mean_ap.update(mini_batch)

    def compute(self):
        """Compute mAP over IOU.
        """
        result = np.mean(
            [
                value
                for ap in self._map_per_iou
                for value in ap.compute().values()
            ]
        )
        return result
