from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from datasetinsights.io.coco import load_coco_annotations
from datasetinsights.stats.visualization.coco_data_visualization import (
    display_ann_for_all_img,
    display_ann_for_single_img,
    display_single_img,
)

parent_dir = Path(__file__).parent.absolute()
mock_coco_dir = parent_dir / "mock_data" / "coco"
coco_ann_file = mock_coco_dir / "annotations" / "keypoint_instances.json"
coco_img_dir = mock_coco_dir / "images"
coco = load_coco_annotations(annotation_file=str(coco_ann_file))


def test_display_single_img():
    mock_ax = Mock()
    mock_fig = Mock()
    mock_subplots = MagicMock(return_value=(mock_fig, mock_ax))

    with patch(
        "datasetinsights.stats.visualization.coco_data_visualization"
        ".plt.subplots",
        mock_subplots,
    ):
        fig = display_single_img(
            coco_obj=coco, img_id=1, data_dir=str(coco_img_dir)
        )
        assert fig == mock_fig


def test_display_ann_for_single_image():
    mock_ax = Mock()
    mock_fig = Mock()
    mock_subplots = MagicMock(return_value=(mock_fig, mock_ax))
    mock_coco_show_ann = Mock()

    with patch(
        "datasetinsights.stats.visualization.coco_data_visualization"
        ".plt.subplots",
        mock_subplots,
    ):
        with patch(
            "datasetinsights.stats.visualization"
            ".coco_data_visualization.COCO.showAnns",
            mock_coco_show_ann,
        ):
            fig = display_ann_for_single_img(
                coco_obj=coco, img_id=1, data_dir=str(coco_img_dir)
            )
            assert mock_coco_show_ann.call_count == 1
            assert fig == mock_fig


def test_display_ann_for_all_img():
    mock_ax_1, mock_ax_2 = Mock(), Mock()
    mock_fig = Mock()
    mock_subplots = MagicMock(return_value=(mock_fig, (mock_ax_1, mock_ax_2)))
    mock_coco_show_ann = Mock()

    with patch(
        "datasetinsights.stats.visualization.coco_data_visualization"
        ".plt.subplots",
        mock_subplots,
    ):
        with patch(
            "datasetinsights.stats.visualization"
            ".coco_data_visualization.COCO.showAnns",
            mock_coco_show_ann,
        ):
            display_ann_for_all_img(
                annotation_file=str(coco_ann_file),
                data_dir=str(coco_img_dir),
                num_imgs=1,
            )

            assert mock_ax_1.imshow.call_count == 1
            assert mock_coco_show_ann.call_count == 1
            annotations = mock_coco_show_ann.call_args_list[0][0][0]
            assert len(annotations) == 8
