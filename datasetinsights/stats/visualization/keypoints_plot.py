""" Helper keypoints library to plot keypoint joints and skeletons  with a
simple Python API.
"""


def get_color_from_color_node(color):
    """ Gets the color from the color node in the template.

    Args:
        color (tuple): The color's channel values expressed in a range from 0..1

    Returns: The color for the node.

    """
    r = int(color["r"] * 255)
    g = int(color["g"] * 255)
    b = int(color["b"] * 255)
    a = int(color["a"] * 255)
    return r, g, b, a


def get_color_for_bone(bone):
    """ Gets the color for the bone from the template.

    Args:
        bone: The active bone.

    Returns: The color of the bone.

    """
    if "color" in bone:
        return get_color_from_color_node(bone["color"])
    else:
        return 255, 0, 255, 255


def get_color_for_keypoint(template, keypoint):
    """ Gets the color for the keypoint from the template.

    Args:
        template: The active template.
        keypoint: The active keypoint.

    Returns: The color for the keypoint.

    """
    node = template["key_points"][keypoint["index"]]

    if "color" in node:
        return get_color_from_color_node(node["color"])
    else:
        return 0, 0, 255, 255


def draw_keypoints_for_figure(image, figure, draw, templates):
    """ Draws keypoints for a figure on an image.

    Args:
        image (PIL Image): a PIL image.
        figure: The figure to draw.
        draw (PIL ImageDraw): PIL image draw interface.
        templates (list): a list of keypoint templates.

    Returns: a PIL image with keypoints for a figure drawn on it.

    """
    # find the template for this
    for template in templates:
        if template["template_id"] == figure["template_guid"]:
            break
    else:
        return image

    # load the spec
    skeleton = template["skeleton"]

    for bone in skeleton:
        j1 = figure["keypoints"][bone["joint1"]]
        j2 = figure["keypoints"][bone["joint2"]]

        if j1["state"] == 2 and j2["state"] == 2:
            x1 = int(j1["x"])
            y1 = int(j1["y"])
            x2 = int(j2["x"])
            y2 = int(j2["y"])

            color = get_color_for_bone(bone)
            draw.line((x1, y1, x2, y2), fill=color, width=6)

    for k in figure["keypoints"]:
        state = k["state"]
        if state == 2:
            x = k["x"]
            y = k["y"]

            color = get_color_for_keypoint(template, k)

            draw.ellipse(
                (x - 3, y - 3, x + 3, y + 3), fill=color, outline=color
            )

    return image
