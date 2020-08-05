r"""Reference.

https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision\
Update algorithm from:
https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
"""

from .average_precision_2d_bbox import AveragePrecisionBBox2D


class AveragePrecision50IOU(AveragePrecisionBBox2D):
    """2D Bounding Box Average Precision at IOU = 50%.

    This would calculate AP@50IOU for each label.
    """

    def __init__(
        self,
    ):
        super().__init__(iou_threshold=0.5)
