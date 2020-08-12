from datasetinsights.data.bbox import BBox2D, group_bbox2d_per_label


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
