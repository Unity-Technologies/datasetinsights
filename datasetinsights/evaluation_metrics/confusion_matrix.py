from .records import Records


def prediction_records(gt_bboxes, pred_bboxes, iou_thresh=0.5):
    """ Calculate prediction results per image.

    Args:
        gt_bboxes (List[BBox2D]): a list of ground truth bounding boxes.
        pred_bboxes (List[BBox2D]): a list of predicted bounding boxes.
        iou_thresh (float): iou threshold. Defaults to 0.5.

    Returns:
        Records: a Records class contains match results.

    """
    records = Records(iou_threshold=iou_thresh)
    records.add_records(gt_bboxes, pred_bboxes)
    return records


def precision_recall(gt_bboxes, pred_bboxes, iou_thresh=0.5):
    """ Calculate precision and recall per image.

    Args:
        gt_bboxes (List[BBox2D]): a list of ground truth bounding boxes.
        pred_bboxes (List[BBox2D]): a list of predicted bounding boxes.
        iou_thresh (float): iou threshold. Defaults to 0.5.

    Returns:
        tuple: (precision_per_image, recall_per_image).

    """
    records = prediction_records(gt_bboxes, pred_bboxes, iou_thresh)
    match_results = records.match_results
    tp_count = 0
    for score, res in match_results:
        if res:
            tp_count += 1
    return tp_count / (len(match_results)), tp_count / len(gt_bboxes)
