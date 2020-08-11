import os

import numpy as np
import torch
from pyquaternion import Quaternion

from datasetinsights.datasets.kitti import (
    KittiBox3d,
    KittiTransforms,
    convert_kitti2nu,
)
from datasetinsights.io.bbox import BBox3d

CUR_FILE = os.path.dirname(os.path.abspath(__file__))


def test_convertpreds2nuscenes():
    """
    in decode batch dimensions are: heatmaps, torch.Size([4, 1, 159, 159])
    pos_offsets,  torch.Size([4, 1, 3, 159, 159])
     dim_offsets, torch.Size([4, 1, 3, 159, 159]) ang_offsets,
     torch.Size([4, 1, 2, 159, 159]) all_grids,
     torch.Size([4, 160, 160, 3])
    Returns:

    """
    t = KittiTransforms(
        os.path.join(CUR_FILE, "..", "tests", "mock_data", "calib000000.txt")
    )
    actual_box = convert_kitti2nu(
        transforms=t,
        bbox=KittiBox3d(
            label="Car",
            position=[0, 0, 0],
            dimensions=[1, 1, 1],
            angle=0,
            sample_idx=-1,
        ),
    )
    expected_box = BBox3d(
        translation=[1, 1, 1],
        size=np.array([1.6569855, 1.728061, 4.3275433], dtype=np.float32),
        rotation=Quaternion(
            0.7065371778859103,
            0.0026742935768291343,
            0.006408422799567937,
            -0.7076418561222197,
        ),
        velocity=np.array([0.0, 0.0, 0.0]),
        label="Car",
        score=torch.tensor(0.6000, dtype=torch.float64),
        sample_token=1,
    )
    assert actual_box.p.all() == expected_box.p.all()
    assert actual_box.rotation == expected_box.rotation
    assert actual_box.size.all() == expected_box.size.all()
    assert (
        np.array(actual_box.translation).all()
        == np.array(expected_box.translation).all()
    )
