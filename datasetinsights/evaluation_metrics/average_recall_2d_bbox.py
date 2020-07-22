r"""Reference.

http://cocodataset.org/#detection-eval
https://arxiv.org/pdf/1502.05082.pdf
https://github.com/rafaelpadilla/Object-Detection-Metrics/issues/22
"""
import collections

from .base import EvaluationMetric
from .records import Records


class AverageRecallBBox2D(EvaluationMetric):
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

            bboxes_per_label = self.label_bboxes(
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

    @staticmethod
    def label_bboxes(pred_bboxes, max_detections):
        """Save bboxes with same label in to a dictionary.

        This operation only apply to predictions for a single image.

        Args:
            pred_bboxes (list): a list of prediction bounding boxes list
            max_detections (int): max detections per label

        Returns:
            labels (dict): a dictionary of prediction boundign boxes
        """
        labels = collections.defaultdict(list)
        for box in pred_bboxes:
            labels[box.label].append(box)
        for label, boxes in labels.items():
            boxes = sorted(boxes, key=lambda bbox: bbox.score, reverse=True)
            # only consider the top confident predictions
            if len(boxes) > max_detections:
                labels[label] = boxes[:max_detections]

        return labels
