TP = True
FP = False


class Records:
    """Save prediction records during update.

    Attributes:
        iou_threshold (float): iou threshold
        match_results (list): save the results (TP/FP)
    Args:
        iou_threshold (float): iou threshold (default: 0.5)
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.match_results = []

    def reset(self):
        self.match_results = []

    def add_records(self, gt_bboxes, pred_bboxes):
        """Add ground truth and prediction records.

        Args:
            gt_bboxes: ground truth bboxes in the current image
            pred_bboxes: sorted predicition bboxes in the current image
        """
        gt_seen = [False] * len(gt_bboxes)

        for pred_bbox in pred_bboxes:
            max_iou = -1
            max_idx = -1
            for i, gt_bbox in enumerate(gt_bboxes):
                if gt_bbox.label != pred_bbox.label:
                    continue
                iou = gt_bbox.iou(pred_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = i
            if max_iou >= self.iou_threshold:
                if not gt_seen[max_idx]:
                    gt_seen[max_idx] = True
                    self.match_results.append((pred_bbox.score, TP))
                else:
                    self.match_results.append((pred_bbox.score, FP))
            else:
                self.match_results.append((pred_bbox.score, FP))
