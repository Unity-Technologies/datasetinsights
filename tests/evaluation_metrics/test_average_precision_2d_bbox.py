from pytest import approx

from datasetinsights.evaluation_metrics import AveragePrecisionBBox2D


def test_average_precision_2d_bbox(get_gt_pred_bbox):
    gt_bboxes, pred_bboxes = get_gt_pred_bbox

    mini_batch1 = [
        [gt_bboxes[0], pred_bboxes[0]],
        [gt_bboxes[1], pred_bboxes[1]],
    ]
    mini_batch2 = [
        [gt_bboxes[2], pred_bboxes[2]],
        [gt_bboxes[3], pred_bboxes[3]],
    ]
    mini_batch3 = [[gt_bboxes[4], pred_bboxes[4]]]

    # test iou threshold = 0.5
    ap_metrics = AveragePrecisionBBox2D(
        iou_start=0.5, iou_end=0.5, iou_step=0.05,
    )
    ap_metrics.update(mini_batch1)
    ap_metrics.update(mini_batch2)
    ap_metrics.update(mini_batch3)

    res = ap_metrics.compute()

    assert approx(res["car"], rel=1e-4) == 0.2257
    assert approx(res["pedestrian"], rel=1e-4) == 0.5
    assert approx(res["bike"], rel=1e-4) == 0

    # test reset function
    ap_metrics.reset()
    res = ap_metrics.compute()
    assert res == {}

    # test iou threshold = 0.5:0.05:0.95
    ap_metrics = AveragePrecisionBBox2D(
        iou_start=0.5, iou_end=0.95, iou_step=0.05,
    )
    ap_metrics.update(mini_batch1)
    ap_metrics.update(mini_batch2)
    ap_metrics.update(mini_batch3)

    res = ap_metrics.compute()
    assert approx(res["car"], rel=1e-4) == 0.08971
    assert approx(res["pedestrian"], rel=1e-4) == 0.26667
    assert approx(res["bike"], rel=1e-4) == 0
