import json


def load_coco_annotations(annotation_file):
    f = open(annotation_file)
    data = json.load(f)
    annotations = data["annotations"]
    return annotations
