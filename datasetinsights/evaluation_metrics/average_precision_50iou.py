r"""Reference.

https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision\
Update algorithm from:
https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
"""

import numpy as np

from .average_precision_2d_bbox import AveragePrecisionBBox2D


class AveragePrecision50IOU(AveragePrecisionBBox2D):
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
    ):
        AveragePrecisionBBox2D.__init__(self, iou_threshold=0.5)
