r"""Reference.

https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision\
Update algorithm from:
https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
"""

import numpy as np

from .average_precision_2d_bbox import AveragePrecisionBBox2D


class MeanAveragePrecision50IOU(AveragePrecisionBBox2D):
    """2D Bounding Box Mean Average Precision metrics at IOU=50%.

    Implementation of classic mAP metrics. We use 10 IoU thresholds
    of .50:.05:.95. This is the same metrics in cocoEval.summarize():
    Average Precision (AP) @[IoU=0.50:0.95 | area = all | maxDets=100]
    """

    def __init__(self):
        super().__init__(iou_threshold=0.5)

    def compute(self):
        """Compute mAP at iou range.
        """
        average_precision = super().compute()
        mean_ap = np.mean(
            [
                result_per_label
                for result_per_label in average_precision.values()
            ]
        )
        return mean_ap
