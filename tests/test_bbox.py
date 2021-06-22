import numpy

from datasetinsights.io.bbox import BBox2D, BBox3D, group_bbox2d_per_label
from datasetinsights.stats.visualization.bbox3d_plot import (
    _project_pt_to_pixel_location,
    _project_pt_to_pixel_location_orthographic,
)


def test_group_bbox2d_per_label():
    count1, count2 = 10, 11
    bbox1 = BBox2D(label="car", x=1, y=1, w=2, h=3)
    bbox2 = BBox2D(label="pedestrian", x=7, y=6, w=3, h=4)
    bboxes = []
    bboxes.extend([bbox1] * count1)
    bboxes.extend([bbox2] * count2)
    bboxes_per_label = group_bbox2d_per_label(bboxes)
    assert len(bboxes_per_label["car"]) == count1
    assert len(bboxes_per_label["pedestrian"]) == count2


def test_group_bbox3d():
    bbox = BBox3D(
        label="na", sample_token=0, translation=[0, 0, 0], size=[5, 5, 5]
    )
    flb = bbox.front_left_bottom_pt
    frb = bbox.front_right_bottom_pt
    flt = bbox.front_left_top_pt
    frt = bbox.front_right_top_pt

    blb = bbox.back_left_bottom_pt
    brb = bbox.back_right_bottom_pt
    blt = bbox.back_left_top_pt
    brt = bbox.back_right_top_pt

    assert flb[0] == flt[0] == blb[0] == blt[0] == -2.5
    assert frb[0] == frt[0] == brb[0] == brt[0] == 2.5

    assert flt[1] == frt[1] == blt[1] == brt[1] == 2.5
    assert flb[1] == frb[1] == blb[1] == brb[1] == -2.5

    assert flt[2] == flb[2] == frt[2] == frb[2] == 2.5
    assert blt[2] == blb[2] == brt[2] == brb[2] == -2.5


def test_project_pt_to_pixel_location():
    pt = [0, 0, 0]
    proj = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    img_height = 480
    img_width = 640

    pixel_loc = _project_pt_to_pixel_location(pt, proj, img_height, img_width)
    assert pixel_loc[0] == 320
    assert pixel_loc[1] == 240

    # more interesting case
    pt = [0, 0, 70]
    proj = numpy.array([[1.299038, 0, 0], [0, 1.7320, 0], [0, 0, -1.0006]])

    pixel_loc = _project_pt_to_pixel_location(pt, proj, img_height, img_width)
    assert pixel_loc[0] == 320
    assert pixel_loc[1] == 240


def test_project_pt_to_pixel_location_orthographic():
    pt = [0, 0, 0]
    proj = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    img_height = 480
    img_width = 640

    pixel_loc = _project_pt_to_pixel_location_orthographic(
        pt, proj, img_height, img_width
    )
    assert pixel_loc[0] == 320
    assert pixel_loc[1] == 240

    # more interesting case
    pt = [0.3, 0, 0]
    proj = numpy.array([[0.08951352, 0, 0], [0, 0.2, 0], [0, 0, -0.0020006]])

    pixel_loc = _project_pt_to_pixel_location_orthographic(
        pt, proj, img_height, img_width
    )
    assert pixel_loc[0] == int(
        (proj[0][0] * pt[0] + 1) * 0.5 * img_width
    )  # 328
    assert pixel_loc[1] == img_width // 2
