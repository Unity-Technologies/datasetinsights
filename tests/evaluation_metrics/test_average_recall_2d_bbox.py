from pytest import approx

from datasetinsights.evaluation_metrics import AverageRecallBBox2D


def test_average_recall_2d_bbox(get_gt_pred_bbox):
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

    # iou_threshold=0.5, max_detections=100
    ar_metrics = AverageRecallBBox2D(iou_threshold=0.5, max_detections=100)
    ar_metrics.update(mini_batch1)
    ar_metrics.update(mini_batch2)
    ar_metrics.update(mini_batch3)
    res = ar_metrics.compute()
    assert approx(res["car"], rel=1e-4) == 0.4
    assert approx(res["pedestrian"], rel=1e-4) == 0.6667
    assert approx(res["bike"], rel=1e-4) == 0

    # test reset function
    ar_metrics.reset()
    res = ar_metrics.compute()
    assert res == {}

    # iou_threshold=0.5, max_detections=1
    ar_metrics.iou_threshold = 0.5
    ar_metrics.max_detections = 1
    ar_metrics.update(mini_batch1)
    ar_metrics.update(mini_batch2)
    ar_metrics.update(mini_batch3)
    res = ar_metrics.compute()
    assert approx(res["car"], rel=1e-4) == 0.3
    assert approx(res["pedestrian"], rel=1e-4) == 0.33333
    assert approx(res["bike"], rel=1e-4) == 0

    ar_metrics.reset()

    # iou_threshold=0.2, max_detections=2
    ar_metrics.iou_threshold = 0.2
    ar_metrics.max_detections = 2
    ar_metrics.update(mini_batch1)
    ar_metrics.update(mini_batch2)
    ar_metrics.update(mini_batch3)
    res = ar_metrics.compute()
    assert approx(res["car"], rel=1e-4) == 0.5
    assert approx(res["pedestrian"], rel=1e-4) == 0.6667
    assert approx(res["bike"], rel=1e-4) == 0
