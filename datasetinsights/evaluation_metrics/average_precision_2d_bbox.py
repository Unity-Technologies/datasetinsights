r"""Reference.

https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision\
Update algorithm from:
https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
"""
import collections

import numpy as np

from .base import EvaluationMetric
from .records import Records


class AveragePrecisionBBox2D(EvaluationMetric):
    """2D Bounding Box Average Precision metrics.

    Attributes:
        label_records (dict): save prediction records for each label
        ap_method (string): AP interoperation method name for AP calculation
        {"EveryPointInterpolation"| "NPointInterpolatedAP"}
        gt_bboxes_count (dict): ground truth box count for each label
        iou_threshold (float): iou threshold

    Args:
        iou_threshold (float): iou threshold (default: 0.5)
        interpolation (string): AP interoperation method name for AP calculation
    """

    def __init__(
        self, iou_threshold=0.5, interpolation="EveryPointInterpolation"
    ):
        if interpolation == "EveryPointInterpolation":
            self.ap_method = self.every_point_interpolated_ap
        elif interpolation == "NPointInterpolatedAP":
            self.ap_method = self.n_point_interpolated_ap
        else:
            raise ValueError(f"Unknown AP method name: {interpolation}!")

        self.iou_threshold = iou_threshold
        self.label_records = collections.defaultdict(
            lambda: Records(iou_threshold=self.iou_threshold)
        )
        self.gt_bboxes_count = collections.defaultdict(int)

    def reset(self):
        """Reset AP metrics."""
        self.label_records = collections.defaultdict(
            lambda: Records(iou_threshold=self.iou_threshold)
        )
        self.gt_bboxes_count = collections.defaultdict(int)

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

            bboxes_per_label = self.label_bboxes(pred_bboxes)
            for label, boxes in bboxes_per_label.items():
                self.label_records[label].add_records(gt_bboxes, boxes)

            for gt_bbox in gt_bboxes:
                self.gt_bboxes_count[gt_bbox.label] += 1

    def compute(self):
        """Compute AP for each label.

        Return:
            average_precision (dict): a dictionary of AP scores per label.
        """
        average_precision = {}
        label_records = self.label_records

        for label in self.gt_bboxes_count:
            # if there are no predicted boxes with this label
            if label not in label_records:
                average_precision[label] = 0
                continue
            pred_infos = label_records[label].pred_infos
            gt_bboxes_count = self.gt_bboxes_count[label]

            pred_infos = sorted(pred_infos, reverse=True)
            true_pos = np.array(list(zip(*pred_infos))[1]).astype(int)
            false_pos = 1 - true_pos

            acc_tp = np.cumsum(true_pos)
            acc_fp = np.cumsum(false_pos)

            recall = acc_tp / gt_bboxes_count
            precision = np.divide(acc_tp, (acc_fp + acc_tp))
            ap = self.ap_method(recall, precision)
            # add class result in the dictionary to be returned
            average_precision[label] = ap

        return average_precision

    @staticmethod
    def label_bboxes(pred_bboxes):
        """Save bboxes with same label in to a dictionary.

        Args:
            pred_bboxes (list): a list of prediction bounding boxes

        Returns:
            labels (dict): a dictionary of prediction bounding boxes
        """
        labels = collections.defaultdict(list)
        for box in pred_bboxes:
            labels[box.label].append(box)
        for label, boxes in labels.items():
            labels[label] = sorted(
                boxes, key=lambda bbox: bbox.score, reverse=True
            )
        return labels

    @staticmethod
    def every_point_interpolated_ap(recall, precision):
        """Calculating the interpolation performed in all points.

        Args:
            recall (list): recall history of the prediction
            precision (list): precision history of the prediction

        Returns: average precision for all points interpolation
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

        Returns: average precision for n-point interpolation
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
