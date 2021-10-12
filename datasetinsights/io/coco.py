import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def load_coco_annotations(annotation_file: str) -> COCO:
    """

    Args:
        annotation_file (str): COCO annotation json file.

    Returns:
        pycocotools.coco.COCO: COCO object

    """
    coco = COCO(annotation_file)
    return coco


def save_figure(fig: plt.Figure, fig_path: str):
    """
    Args:
        fig (plt.Figure): Figure object
        fig_path (str): Path where figure is to be saved
    """
    fig.savefig(fname=fig_path, bbox_inches="tight", pad_inches=0.15)
