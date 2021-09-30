import json


def load_coco_annotations(json_file):
    f = open(json_file)
    data = json.load(f)
    annotations = data["annotations"]
    return annotations
