from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from datasetinsights.stats.visualization.coco_data_visualization import (
    display_ann_for_multiple_img,
    display_single_img,
)

parent_dir = Path(__file__).parent.absolute()
mock_coco_dir = parent_dir / "mock_data" / "coco"
coco_ann_file = mock_coco_dir / "annotations" / "keypoints.json"
coco_img_dir = mock_coco_dir / "images"


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
            annotation_file=str(coco_ann_file),
            img_id=61855733451949387398181790757513827492,
            data_dir=str(coco_img_dir),
        )
        assert fig == mock_fig


def test_display_single_img_with_ann():
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
            fig = display_single_img(
                annotation_file=str(coco_ann_file),
                img_id=61855733451949387398181790757513827492,
                data_dir=str(coco_img_dir),
                show_annotation=True,
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
            display_ann_for_multiple_img(
                annotation_file=str(coco_ann_file),
                data_dir=str(coco_img_dir),
                num_imgs=1,
            )

            assert mock_ax_1.imshow.call_count == 1
            assert mock_coco_show_ann.call_count == 1
