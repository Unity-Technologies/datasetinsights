import pytest

from datasetinsights.io.bbox import BBox2D


@pytest.fixture
def get_gt_pred_bbox():
    gt_bbox1 = BBox2D(label="car", x=1, y=1, w=2, h=3)
    gt_bbox2 = BBox2D(label="car", x=7, y=6, w=3, h=4)
    gt_bbox11 = BBox2D(label="pedestrian", x=1, y=6, w=2, h=4)
    gt_bbox3 = BBox2D(label="car", x=2, y=2, w=2, h=2)
    gt_bbox4 = BBox2D(label="car", x=2, y=6, w=2, h=4)
    gt_bbox5 = BBox2D(label="car", x=6, y=5, w=4, h=3)
    gt_bbox14 = BBox2D(label="bike", x=6, y=1, w=3, h=2)
    gt_bbox6 = BBox2D(label="car", x=2, y=1, w=2, h=3)
    gt_bbox7 = BBox2D(label="car", x=6, y=3, w=3, h=5)
    gt_bbox8 = BBox2D(label="car", x=2, y=1, w=5, h=2)
    gt_bbox9 = BBox2D(label="car", x=2, y=4, w=3, h=4)
    gt_bbox10 = BBox2D(label="car", x=5, y=1, w=5, h=4)
    gt_bbox12 = BBox2D(label="pedestrian", x=1, y=5, w=3, h=4)
    gt_bbox13 = BBox2D(label="pedestrian", x=8, y=7, w=2, h=2)

    pred_bbox1 = BBox2D(label="car", x=1, y=2, w=3, h=3, score=0.93)
    pred_bbox2 = BBox2D(label="car", x=6, y=5, w=3, h=4, score=0.94)
    pred_bbox13 = BBox2D(label="pedestrian", x=1, y=6, w=2, h=3, score=0.70)
    pred_bbox16 = BBox2D(label="pedestrian", x=1, y=7, w=2, h=3, score=0.80)
    pred_bbox3 = BBox2D(label="car", x=2, y=5, w=2, h=4, score=0.79)
    pred_bbox4 = BBox2D(label="car", x=5, y=4, w=4, h=2, score=0.39)
    pred_bbox5 = BBox2D(label="car", x=5, y=7, w=4, h=2, score=0.49)
    pred_bbox6 = BBox2D(label="car", x=2, y=2, w=2, h=2, score=0.59)
    pred_bbox7 = BBox2D(label="car", x=2, y=6, w=2, h=2, score=0.69)
    pred_bbox8 = BBox2D(label="car", x=6, y=3, w=4, h=4, score=0.79)
    pred_bbox9 = BBox2D(label="car", x=1, y=1, w=7, h=2, score=0.99)
    pred_bbox10 = BBox2D(label="car", x=4, y=5, w=3, h=4, score=0.90)
    pred_bbox11 = BBox2D(label="car", x=1, y=1, w=2, h=3, score=0.80)
    pred_bbox12 = BBox2D(label="car", x=4, y=4, w=5, h=2, score=0.70)
    pred_bbox14 = BBox2D(label="pedestrian", x=3, y=7, w=3, h=3, score=0.40)
    pred_bbox15 = BBox2D(label="pedestrian", x=8, y=7, w=2, h=3, score=0.30)

    gt_bboxes = [
        [gt_bbox1, gt_bbox2, gt_bbox11],
        [gt_bbox3, gt_bbox4, gt_bbox5, gt_bbox14],
        [gt_bbox6, gt_bbox7],
        [gt_bbox8, gt_bbox9],
        [gt_bbox10, gt_bbox12, gt_bbox13],
    ]

    pred_bboxes = [
        [pred_bbox1, pred_bbox2, pred_bbox13, pred_bbox16],
        [pred_bbox3, pred_bbox4, pred_bbox5],
        [pred_bbox6, pred_bbox7, pred_bbox8],
        [pred_bbox9, pred_bbox10],
        [pred_bbox11, pred_bbox12, pred_bbox14, pred_bbox15],
    ]

    return gt_bboxes, pred_bboxes
