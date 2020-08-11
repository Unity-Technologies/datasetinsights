import logging
import operator
from typing import Dict, List, Tuple

import numpy as np
from nuscenes.eval.detection.data_classes import DetectionMetrics

from datasetinsights.io.bbox import BBox3d

from .average_precision_config import (
    DIST_FCN,
    DIST_THS,
    LABEL_RANGE,
    MAX_BOXES_PER_SAMPLE,
    MEAN_AP_WEIGHT,
    MIN_PRECISION,
    MIN_RECALL,
)
from .base import EvaluationMetric

logger = logging.getLogger(__name__)


class DetectionConfig:
    """ Data class that specifies the detection evaluation settings. """

    def __init__(
        self,
        class_range: Dict[str, int] = LABEL_RANGE,
        dist_fcn: str = DIST_FCN,
        dist_ths: List[float] = DIST_THS,
        min_recall: float = MIN_RECALL,
        min_precision: float = MIN_PRECISION,
        max_boxes_per_sample: float = MAX_BOXES_PER_SAMPLE,
        mean_ap_weight: int = MEAN_AP_WEIGHT,
    ):
        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.labels = self.class_range.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            "class_range": self.class_range,
            "dist_fcn": self.dist_fcn,
            "dist_ths": self.dist_ths,
            "min_recall": self.min_recall,
            "min_precision": self.min_precision,
            "max_boxes_per_sample": self.max_boxes_per_sample,
            "mean_ap_weight": self.mean_ap_weight,
        }

    @classmethod
    def deserialize(cls, content):
        """ Initialize from serialized dictionary. """
        return cls(
            content["class_range"],
            content["dist_fcn"],
            content["dist_ths"],
            content["min_recall"],
            content["min_precision"],
            content["max_boxes_per_sample"],
            content["mean_ap_weight"],
        )


def calc_ap(*, precision, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    # assert 0 <= min_precision < 1
    # assert 0 <= min_recall <= 1

    prec = np.copy(precision)
    prec = prec[
        round(100 * min_recall) + 1 :
    ]  # Clip low recalls. +1 to exclude the min
    # recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def center_distance(*, gt_box: BBox3d, pred_box: BBox3d) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(
        np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2])
    )


def _count_class_examples(
    boxes: Dict[str, Tuple[List[BBox3d], List[BBox3d]]], label
):
    npos = 0
    for image, (predicted_bbs, gt_bbs) in boxes.items():
        for gt_bounding_box in gt_bbs:
            if gt_bounding_box.label == label:
                npos += 1
    return npos


class AveragePrecision(EvaluationMetric):
    def __init__(self, config: DetectionConfig = None):
        self._num_examples = 0
        self.cfg = config or DetectionConfig()
        self.boxes = {}

    # todo need update method which accumulates serialized boxes
    def update(self, boxes):
        self.boxes.update(boxes)

    def reset(self):
        self.boxes = {}

    def _calculate_label_ap_simple(
        self,
        boxes: Dict[str, Tuple[List[BBox3d], List[BBox3d]]],
        label,
        dist_th,
    ):
        """
            calculate the average precision for a single class at a single
            distance threshold.
            :param boxes: keys -> (list of predicted boxes, list of ground
            truth boxes)
            :param label: name of the class used to calculate ap
            :param dist_th: distance threshold. Distance from center of the
            predicted box using birds-eye-view to
            ground truth box's center using birds eye view
            :return: average precesion for a single class at a single distance
             threshold
            """
        tp = []  # Accumulator of true positives
        fp = []  # Accumulator of false positives
        npos = _count_class_examples(boxes=boxes, label=label)
        for img, (predicted_bbs, gt_bbs) in boxes.items():
            predicted_bbs.sort(key=operator.attrgetter("score"), reverse=True)
            matches = set()
            for predicted_bb_index, predicted_box in enumerate(predicted_bbs):
                min_dist = np.inf
                for gt_idx, gt_box in enumerate(gt_bbs):
                    if gt_box.label == label and gt_idx not in matches:
                        this_distance = center_distance(
                            gt_box=gt_box, pred_box=predicted_box
                        )
                        if this_distance < min_dist:
                            min_dist = this_distance
                    is_match = min_dist < dist_th
                    if is_match:
                        tp.append(1)
                        fp.append(0)
                    else:
                        tp.append(0)
                        fp.append(1)
        if np.max(tp) == 0:
            return 0
        else:
            tp = np.cumsum(tp).astype(np.float)
            fp = np.cumsum(fp).astype(np.float)
            prec = tp / (fp + tp)
            rec = tp / float(npos)
            rec_interp = np.linspace(
                0, 1, 101
            )  # 101 steps, from 0% to 100% recall.
            prec = np.interp(rec_interp, rec, prec, right=0)
        ap = calc_ap(
            precision=prec,
            min_recall=self.cfg.min_recall,
            min_precision=self.cfg.min_precision,
        )
        return ap

    def compute(
        self, boxes: Dict[str, Tuple[List[BBox3d], List[BBox3d]]] = None
    ) -> Dict[str, float]:
        """
        calculate the mean ap for all classes over all distance thresholds
        defined in the config. The equation is
        described in the second figure in the nuscenes paper
        https://arxiv.org/pdf/1903.11027.pdf It is the normalized
        sum of the ROC curves for each class and distance.
        :param boxes: the predicted and ground truth bounding boxes per sample.
        :return: dictionary mapping each label to it's average precision
        (averaged across all distance thresholds for
        that label)
        """
        if boxes is None:
            boxes = self.boxes
        metrics = DetectionMetrics(self.cfg)
        for label in self.cfg.labels:
            for dist_th in self.cfg.dist_ths:
                ap = self._calculate_label_ap_simple(
                    boxes=boxes, dist_th=dist_th, label=label
                )
                metrics.add_label_ap(label, dist_th, ap)
        return metrics.mean_dist_aps
