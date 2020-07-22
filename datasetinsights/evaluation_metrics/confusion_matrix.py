from .records import Records


def prediction_records(gt_bboxes, pred_bboxes, iou_thresh=0.5):
    """ Calculate prediction results per image.

    Args:
        gt_bboxes (List[BBox2D]): a list of ground truth bounding boxes.
        pred_bboxes (List[BBox2D]): a list of predicted bounding boxes.
        iou_thresh (float): iou threshold. Defaults to 0.5.

    Returns (list):
        A list of tuple of prediction results, e.g. [(score1, TP), (score, FP)].

    """
    records = Records(iou_threshold=iou_thresh)
    records.add_records(gt_bboxes, pred_bboxes)
    return records.pred_infos


def precision_recall(gt_bboxes, pred_bboxes, iou_thresh=0.5):
    """ Calculate precision and recall per image.

    Args:
        gt_bboxes (List[BBox2D]): a list of ground truth bounding boxes.
        pred_bboxes (List[BBox2D]): a list of predicted bounding boxes.
        iou_thresh (float): iou threshold. Defaults to 0.5.

    Returns (tuple):
        a tuple. (precision_per_image, recall_per_image).

    """
    records = prediction_records(gt_bboxes, pred_bboxes, iou_thresh)
    tp_count = 0
    for score, res in records:
        if res:
            tp_count += 1
    return tp_count / (len(records)), tp_count / len(gt_bboxes)
