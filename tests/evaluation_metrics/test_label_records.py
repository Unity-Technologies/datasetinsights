from datasetinsights.evaluation_metrics.records import Records
from datasetinsights.io.bbox import BBox2D


def get_gt_pred_bbox():
    gt_bbox1 = BBox2D(label="car", x=1, y=1, w=2, h=3)
    gt_bbox2 = BBox2D(label="car", x=7, y=6, w=3, h=4)
    gt_bbox3 = BBox2D(label="car", x=2, y=6, w=2, h=4)

    pred_bbox1 = BBox2D(label="car", x=1, y=2, w=3, h=3, score=0.93)
    pred_bbox2 = BBox2D(label="car", x=6, y=5, w=3, h=4, score=0.94)
    pred_bbox3 = BBox2D(label="car", x=2, y=5, w=2, h=4, score=0.79)

    gt_bboxes = [gt_bbox1, gt_bbox2, gt_bbox3]

    pred_bboxes = [pred_bbox1, pred_bbox2, pred_bbox3]

    return gt_bboxes, pred_bboxes


def test_label_records():
    gt_bboxes, pred_bboxes = get_gt_pred_bbox()
    bboxes = [
        [gt_bboxes[0], pred_bboxes[0]],
        [gt_bboxes[1], pred_bboxes[1]],
        [gt_bboxes[2], pred_bboxes[2]],
    ]

    records = Records(iou_threshold=0.5)
    for bbox in bboxes:
        gt_bbox, pred_bbox = bbox
        records.add_records([gt_bbox], [pred_bbox])

    tp_count = sum(list(zip(*records.pred_infos))[1])
    assert tp_count == 1

    records.reset()
    assert records.pred_infos == []
