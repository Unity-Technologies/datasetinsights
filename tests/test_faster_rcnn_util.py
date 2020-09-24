"""unit test case for frcnn util."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from datasetinsights.estimators.faster_rcnn import (
    FasterRCNN,
    _gt_preds2tensor,
    canonical2list,
    convert_bboxes2canonical,
    gather_gt_preds,
    list2canonical,
    list3d_2canonical,
    metric_per_class_plot,
    pad_box_lists,
    prepare_bboxes,
    reduce_dict,
    tensorlist2canonical,
)
from datasetinsights.io.bbox import BBox2D

padding_box = BBox2D(
    label=np.nan, score=np.nan, x=np.nan, y=np.nan, w=np.nan, h=np.nan
)


def test_pad_box_lists():
    """test pad box lists."""
    box_a, box_b = (
        BBox2D(label=0, x=10, y=10, w=10, h=10),
        BBox2D(label=1, x=20, y=20, w=10, h=10),
    )
    uneven_list = [
        ([box_a], []),
        ([box_a, box_b], [box_b]),
        ([box_b], [box_a, box_b]),
        ([box_b], [box_a]),
    ]

    actual_result = pad_box_lists(uneven_list, max_boxes_per_img=3)
    expected_result = [
        (
            [box_a, padding_box, padding_box],
            [padding_box, padding_box, padding_box],
        ),
        ([box_a, box_b, padding_box], [box_b, padding_box, padding_box]),
        ([box_b, padding_box, padding_box], [box_a, box_b, padding_box]),
        ([box_b, padding_box, padding_box], [box_a, padding_box, padding_box]),
    ]
    for i in range(len(expected_result)):
        assert len(expected_result[i][0]) == len(actual_result[i][0])
        assert len(expected_result[i][1]) == len(actual_result[i][1])
        for t_index in range(2):
            for j in range(len(expected_result[i][t_index])):
                if np.isnan(expected_result[i][t_index][j].label):
                    assert np.isnan(actual_result[i][t_index][j].label)
                else:
                    assert (
                        expected_result[i][t_index][j]
                        == actual_result[i][t_index][j]
                    )
    assert True


def test_list3d_2canonical():
    """test list 3d  to canonical."""
    box_a, box_b = (
        BBox2D(label=0, x=10, y=10, w=10, h=10),
        BBox2D(label=1, x=20, y=20, w=10, h=10),
    )
    list3d = [
        [
            [
                [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        ],
        [
            [
                [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        ],
        [
            [
                [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        ],
        [
            [
                [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        ],
    ]
    expected_result = [
        ([box_a], []),
        ([box_a, box_b], [box_b]),
        ([box_b], [box_a, box_b]),
        ([box_b], [box_a]),
    ]
    actual_result = list3d_2canonical(list3d)
    assert actual_result == expected_result


def test_gt_preds2tensor():
    """test prediction to tensor conversion."""
    box_a, box_b = (
        BBox2D(label=0, x=10, y=10, w=10, h=10),
        BBox2D(label=1, x=20, y=20, w=10, h=10),
    )
    uneven_list = [
        ([box_a], []),
        ([box_a, box_b], [box_b]),
        ([box_b], [box_a, box_b]),
        ([box_b], [box_a]),
    ]
    actual_result = _gt_preds2tensor(uneven_list, 3)
    expected_result = torch.Tensor(
        [
            [
                [
                    [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            [
                [
                    [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            [
                [
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            [
                [
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
        ]
    )
    torch.eq(expected_result, actual_result)


def test_convert_empty():
    """test convert empty."""
    targets = prepare_bboxes([])
    assert len(targets["boxes"]) < 1


def _same_dict(expected, actual):
    """test same dict."""
    assert len(expected.keys()) == len(actual.keys())
    for k in expected.keys():
        expected_tensor = expected[k]
        actual_tensor = actual[k]
        assert torch.all(torch.eq(expected_tensor, actual_tensor))
    return True


def _same_dict_list(expected, actual):
    """test same dict list."""
    assert len(expected) == len(actual)
    for i in range(len(expected)):
        _same_dict(expected[i], actual[i])
    return True


def test_convert2torchvision_format():
    """test convert  to torchvision format."""
    boxes = [
        BBox2D(label=0, x=10, y=10, w=10, h=10),
        BBox2D(label=1, x=20, y=20, w=10, h=10),
    ]

    actual_targets = prepare_bboxes(boxes)
    expected_targets = {
        "boxes": torch.Tensor([[10, 10, 20, 20], [20, 20, 30, 30]]),
        "labels": torch.LongTensor([0, 1]),
    }

    assert _same_dict(expected_targets, actual_targets)


def same_list_of_list_of_bboxes(l_1, l_2):
    """test same list of list of bboxes."""
    assert len(l_1) == len(l_2)
    for i in range(len(l_1)):
        assert len(l_1[i]) == len(l_2[i])
        for j in range(len(l_1[i])):
            assert l_1[i][j] == l_2[i][j]
    return True


def test_convert2canonical():
    """test convert to canonical."""
    boxes_rcnn_format = [
        {
            "boxes": torch.Tensor(
                [[10.5, 10.5, 20.5, 20.5], [20.5, 20.5, 30.5, 30.5]]
            ),
            "labels": torch.Tensor([0, 1]),
            "scores": torch.FloatTensor([0.3, 0.9]),
        }
    ]
    actual_result = convert_bboxes2canonical(boxes_rcnn_format)
    expected_result = [
        [
            BBox2D(label=0, x=10.5, y=10.5, w=10, h=10, score=0.3),
            BBox2D(label=1, x=20.5, y=20.5, w=10, h=10, score=0.9),
        ]
    ]
    assert same_list_of_list_of_bboxes(actual_result, expected_result)


def test_convert2canonical_batch():
    """test convert to canonical batch."""
    boxes_rcnn_format = [
        {
            "boxes": torch.Tensor([[10.0, 10, 20, 20], [20, 20, 30, 30]]),
            "labels": torch.LongTensor([0, 1]),
        },
        {
            "boxes": torch.Tensor([[10, 10, 20, 20], [20, 20, 30, 30]]),
            "labels": torch.LongTensor([2, 3]),
        },
    ]
    actual_result = convert_bboxes2canonical(boxes_rcnn_format)
    expected_result = [
        [
            BBox2D(label=0, x=10, y=10, w=10, h=10),
            BBox2D(label=1, x=20, y=20, w=10, h=10),
        ],
        [
            BBox2D(label=2, x=10, y=10, w=10, h=10),
            BBox2D(label=3, x=20, y=20, w=10, h=10),
        ],
    ]
    assert same_list_of_list_of_bboxes(actual_result, expected_result)


@patch("datasetinsights.estimators.faster_rcnn.get_world_size")
def test_reduce_dict_non_dist(test_patch):
    """test reduce dict non dist."""
    input_dict = {
        "loss_classifier": np.nan,
        "loss_box_reg": np.nan,
        "loss_objectness": np.nan,
        "loss_rpn_box_reg": np.nan,
    }
    test_patch.return_value = 1
    expected_result = reduce_dict(input_dict)
    assert input_dict == expected_result


@patch("datasetinsights.estimators.faster_rcnn.get_world_size")
@patch("datasetinsights.estimators.faster_rcnn.dist.all_reduce")
def test_reduce_dict_dist(mock_all_reduce, mock_get_world_size):
    """test reduce dict dist."""
    input_dict = {
        "loss_classifier": torch.Tensor([0.4271]),
        "loss_box_reg": torch.Tensor([0.4271]),
        "loss_objectness": torch.Tensor([0.4271]),
        "loss_rpn_box_reg": torch.Tensor([0.4271]),
    }
    mock_get_world_size.return_value = 3
    mock_all_reduce.return_value = MagicMock()

    actual_result = reduce_dict(input_dict)
    expected_result = {
        "loss_classifier": 0.1423666626214981,
        "loss_box_reg": 0.1423666626214981,
        "loss_objectness": 0.1423666626214981,
        "loss_rpn_box_reg": 0.1423666626214981,
    }

    assert all(
        [
            actual_result["loss_classifier"].item()
            == expected_result["loss_classifier"]
            and actual_result["loss_box_reg"].item()
            == expected_result["loss_box_reg"]
            and actual_result["loss_classifier"].item()
            == expected_result["loss_classifier"]
            and actual_result["loss_objectness"].item()
            == expected_result["loss_objectness"]
            and actual_result["loss_rpn_box_reg"].item()
            == expected_result["loss_rpn_box_reg"]
        ]
    )


def test_canonical2list():
    """test canonical to list."""
    bbox = BBox2D(label=0, x=10, y=10, w=10, h=10)
    actual_result = canonical2list(bbox)
    expected_result = [0, 1.0, 10, 10, 10, 10]
    assert actual_result == expected_result


def test_list2canonical():
    """test list to canonical."""
    input_list = [0, 1.0, 10, 10, 10, 10]
    actual_result = list2canonical(input_list)

    assert all(
        [
            actual_result.label == input_list[0]
            and actual_result.score == input_list[1]
            and actual_result.x == input_list[2]
            and actual_result.y == input_list[3]
            and actual_result.w == input_list[4]
            and actual_result.h == input_list[5]
        ]
    )


@patch("datasetinsights.estimators.faster_rcnn.get_world_size")
@patch("datasetinsights.estimators.faster_rcnn.dist.all_gather")
def test_gather_gt_preds(mock_all_gather, mock_get_world_size):
    """test gather preds."""
    box_a, box_b = (
        BBox2D(label=0, x=10, y=10, w=10, h=10),
        BBox2D(label=1, x=20, y=20, w=10, h=10),
    )
    uneven_list = [
        ([box_a], []),
        ([box_a, box_b], [box_b]),
        ([box_b], [box_a, box_b]),
        ([box_b], [box_a]),
    ]
    mock_get_world_size.return_value = 1
    mock_all_gather.return_value = MagicMock()
    actual_result = gather_gt_preds(
        gt_preds=uneven_list, device=torch.device("cpu"), max_boxes=3
    )
    assert len(actual_result) == 4


def test_tensorlist2canonical():
    """test tensor list to canonical."""
    input_tensor = torch.Tensor(
        [
            [
                [
                    [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            [
                [
                    [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            [
                [
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            [
                [
                    [1.0, 1.0, 20.0, 20.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [0.0, 1.0, 10.0, 10.0, 10.0, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
        ]
    )
    tensor_list = [torch.empty(list(input_tensor.size()))]
    actual_result = tensorlist2canonical(tensor_list)

    assert len(actual_result) == 4


@patch("datasetinsights.estimators.faster_rcnn.plt")
def test_metric_per_class_plot(mock_plt):
    """test metric per class plot."""
    label_mappings = {
        "0": "",
        "1": "book_dorkdiaries_aladdin",
        "2": "candy_minipralines_lindt",
        "3": "candy_raffaello_confetteria",
        "4": "cereal_capn_crunch",
        "5": "cereal_cheerios_honeynut",
        "6": "cereal_corn_flakes",
        "7": "cereal_cracklinoatbran_kelloggs",
        "8": "cereal_oatmealsquares_quaker",
        "9": "cereal_puffins_barbaras",
        "10": "cereal_raisin_bran",
        "11": "cereal_rice_krispies",
        "12": "chips_gardensalsa_sunchips",
        "13": "chips_sourcream_lays",
        "14": "cleaning_freegentle_tide",
    }
    data = {"1": 0.1, "2": 0.2, "9": 0.3, "11": 0.4}

    metric_per_class_plot("loss", data, label_mappings, figsize=(20, 10))
    mock_plt.title.assert_called_once_with("loss per class")
    mock_plt.bar.assert_called_once_with(
        ["1", "11", "2", "9"], [0.1, 0.4, 0.2, 0.3]
    )
    assert mock_plt.figure.called


def test_collate_fn():
    """test collate fn."""
    input_tupple = (("x0", "x1", "x2", "xn"), ("y0", "y1", "y2", "yn"))
    actual_result = FasterRCNN.collate_fn(input_tupple)
    expected_result = (("x0", "y0"), ("x1", "y1"), ("x2", "y2"), ("xn", "yn"))
    assert actual_result == expected_result
