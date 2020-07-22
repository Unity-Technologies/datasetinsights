from pytest import approx

from datasetinsights.evaluation_metrics.confusion_matrix import (
    precision_recall,
    prediction_records,
)


def test_prediction_records(get_gt_pred_bbox):
    gt_bboxes, pred_bboxes = get_gt_pred_bbox

    img1_gt_bboxes, img1_pred_bboxes = gt_bboxes[0], pred_bboxes[0]
    img2_gt_bboxes, img2_pred_bboxes = gt_bboxes[1], pred_bboxes[1]

    # test iou threshold = 0.5
    pred_info1 = prediction_records(
        img1_gt_bboxes, img1_pred_bboxes, iou_thresh=0.5
    )
    true_res1 = [False, False, True, False]
    pred_info2 = prediction_records(
        img2_gt_bboxes, img2_pred_bboxes, iou_thresh=0.5
    )
    true_res2 = [True, False, False]

    for i in range(len(pred_info1)):
        assert pred_info1[i][0] == img1_pred_bboxes[i].score
        assert pred_info1[i][1] == true_res1[i]

    for i in range(len(pred_info2)):
        assert pred_info2[i][0] == img2_pred_bboxes[i].score
        assert pred_info2[i][1] == true_res2[i]

    # test iou threshold = 0.3
    pred_info1 = prediction_records(
        img1_gt_bboxes, img1_pred_bboxes, iou_thresh=0.3
    )
    true_res1 = [True, True, True, False]
    pred_info2 = prediction_records(
        img2_gt_bboxes, img2_pred_bboxes, iou_thresh=0.3
    )
    true_res2 = [True, False, False]

    for i in range(len(pred_info1)):
        assert pred_info1[i][0] == img1_pred_bboxes[i].score
        assert pred_info1[i][1] == true_res1[i]

    for i in range(len(pred_info2)):
        assert pred_info2[i][0] == img2_pred_bboxes[i].score
        assert pred_info2[i][1] == true_res2[i]


def test_precision_recall(get_gt_pred_bbox):
    gt_bboxes, pred_bboxes = get_gt_pred_bbox

    img1_gt_bboxes, img1_pred_bboxes = gt_bboxes[0], pred_bboxes[0]
    img2_gt_bboxes, img2_pred_bboxes = gt_bboxes[1], pred_bboxes[1]

    # test iou threshold = 0.5
    precision1, recall1 = precision_recall(
        img1_gt_bboxes, img1_pred_bboxes, iou_thresh=0.5
    )
    precision2, recall2 = precision_recall(
        img2_gt_bboxes, img2_pred_bboxes, iou_thresh=0.5
    )

    assert approx(precision1, rel=1e-2) == 0.25
    assert approx(recall1, rel=1e-2) == 0.33
    assert approx(precision2, rel=1e-2) == 0.33
    assert approx(recall2, rel=1e-2) == 0.25

    # test iou threshold = 0.3
    precision1, recall1 = precision_recall(
        img1_gt_bboxes, img1_pred_bboxes, iou_thresh=0.3
    )
    precision2, recall2 = precision_recall(
        img2_gt_bboxes, img2_pred_bboxes, iou_thresh=0.3
    )

    assert approx(precision1, rel=1e-2) == 0.75
    assert approx(recall1, rel=1e-2) == 1.0
    assert approx(precision2, rel=1e-2) == 0.33
    assert approx(recall2, rel=1e-2) == 0.25
