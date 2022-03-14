import json
import pathlib

from datasetinsights.stats.image_analysis import (
    get_fg_bg_var_laplacian,
    laplacian_img,
)


def test_get_fg_bg_var_laplacian():
    cur_dir = pathlib.Path(__file__).parent.absolute()
    img_path = str(
        cur_dir
        / "mock_data"
        / "coco"
        / "images"
        / "camera_61855733451949387398181790757513827492.png"
    )
    ann_path = str(
        cur_dir / "mock_data" / "coco" / "annotations" / "keypoints.json"
    )
    laplacian = laplacian_img(img_path)
    f = open(ann_path)
    annotations = json.load(f)["annotations"]
    bbox_var_lap, img_var_lap = get_fg_bg_var_laplacian(laplacian, annotations)
    assert len(bbox_var_lap) > 0
    assert len(img_var_lap) > 0
