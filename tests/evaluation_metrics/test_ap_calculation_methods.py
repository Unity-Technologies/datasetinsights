from pytest import approx

from datasetinsights.evaluation_metrics import AveragePrecision


def get_precision_recall():
    precision = [
        1,
        0.5,
        0.6666,
        0.5,
        0.4,
        0.3333,
        0.2857,
        0.25,
        0.2222,
        0.3,
        0.2727,
        0.3333,
        0.3846,
        0.4285,
        0.4,
        0.375,
        0.3529,
        0.3333,
        0.3157,
        0.3,
        0.2857,
        0.2727,
        0.3043,
        0.2916,
    ]
    recall = [
        0.0666,
        0.0666,
        0.1333,
        0.1333,
        0.1333,
        0.1333,
        0.1333,
        0.1333,
        0.1333,
        0.2,
        0.2,
        0.2666,
        0.3333,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4666,
        0.4666,
    ]

    return precision, recall


def test_average_precision_2d_bbox():
    precision, recall = get_precision_recall()

    n_points_inter1 = AveragePrecision.n_point_interpolated_ap(
        recall, precision, point=11
    )
    n_points_inter2 = AveragePrecision.n_point_interpolated_ap(
        recall, precision, point=6
    )
    n_points_inter3 = AveragePrecision.n_point_interpolated_ap(
        recall, precision, point=101
    )
    all_points_inter = AveragePrecision.every_point_interpolated_ap(
        recall, precision
    )

    assert approx(n_points_inter1, rel=1e-3) == 0.2684
    assert approx(n_points_inter2, rel=1e-3) == 0.3095
    assert approx(n_points_inter3, rel=1e-3) == 0.2481
    assert approx(all_points_inter, rel=1e-3) == 0.2456
