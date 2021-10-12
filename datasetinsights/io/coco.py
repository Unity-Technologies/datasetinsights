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
