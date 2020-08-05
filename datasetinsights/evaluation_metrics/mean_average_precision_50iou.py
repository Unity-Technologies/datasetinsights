r"""Reference.

https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision\
Update algorithm from:
https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
"""

import numpy as np

from .average_precision_2d_bbox import AveragePrecisionBBox2D


class MeanAveragePrecision50IOU(AveragePrecisionBBox2D):
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

    def __init__(self,):
        AveragePrecisionBBox2D.__init__(self, iou_threshold=0.5)

    def compute(self):
        """Compute AP for each label.

        Return:
            mAP (float): mean average precision across all ious
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
        mean_ap = np.mean(
            [
                result_per_label
                for result_per_label in average_precision.values()
            ]
        )
        return mean_ap
