import random

import cv2
import numpy as np
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Flip the image from top to bottom.

    Args:
        flip_prob: the probability to flip the image
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, depth = sample
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            depth = F.hflip(depth)

        return (image, depth)


class Resize:
    """Resize the (image, target) to the given sizes.

    Args:
        img_size (tuple or int): Desired output size. If size is a
            sequence like (h, w), output size will be matched to this. If size
            is an int, smaller edge of the image will be matched to this
            number. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        target_size (tuple or int): Desired output size. If size is a
            sequence like (h, w), output size will be matched to this. If size
            is an int, smaller edge of the image will be matched to this
            number. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    def __init__(self, img_size=-1, target_size=-1):
        self.img_size = img_size
        self.target_size = target_size

    def __call__(self, sample):
        img, target = sample
        if self.img_size != -1:
            img = F.resize(img, self.img_size)
        if self.target_size != -1:
            target = F.resize(target, self.target_size)
        return (img, target)


class BlurImage:
    """Blur the image.

    Args:
        None
    """

    def __init__(self, p):
        self.p = p

    def rand_kernel(self):
        size = np.random.randn(1)
        size = int(np.round(size)) * 2 + 1
        if size < 0:
            return None
        if random.random() < 0.5:
            return None
        size = min(size, 45)
        kernel = np.zeros((size, size))
        c = int(size / 2)
        wx = random.random()
        kernel[:, c] += 1.0 / size * wx
        kernel[c, :] += 1.0 / size * (1 - wx)
        return kernel

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample[0]
            kernel = self.rand_kernel()

            if kernel is not None:
                image = cv2.filter2D(image, -1, kernel)
            return image, None

        return sample[0], None


class GrayImage:
    """ Convert image into grayscale

    Args:
        None
    Return:
        3-channel grayscale image
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, _ = sample[0], sample[1]
        if random.random() < self.p:
            grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.zeros((grayed.shape[0], grayed.shape[1], 3), np.uint8)
            image[:, :, 0] = image[:, :, 1] = image[:, :, 2] = grayed
            return image
        return image, None


class CenterCrop:
    """ Returns a crop of image with required size
    Init:
        Size of the crop
    Args:
        Image to be croppped passed as tuple (image,_)
    Returns:
        Center crop with init size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample[0]
        shape = img.shape[1]
        if shape == self.size:
            return img
        c = shape // 2
        # noqa: E741 IDK why it says ambiguous
        l = c - self.size // 2
        r = c + self.size // 2 + 1
        return img[l:r, l:r], None


class ShiftBBox:
    def __init__(self, helper, shift):
        self.helper = helper
        self.shift = shift

    def __call__(self, bbox, shape):
        shift_val = (
            self.helper["random_number"](-1, 1) * self.shift,
            self.helper["random_number"](-1, 1) * self.shift,
        )
        x1, y1, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        shift_x, shift_y = shift_val[0], shift_val[1]
        # corner form and center form created
        # calculate corner
        x2 = x1 + w
        y2 = y1 + h
        # calculate center
        # x, y = x1 + w // 2, y1 + h // 2
        imh, imw = shape[:2]

        shift_x = max(-x1, min(imw - 1 - x2, shift_x))
        shift_y = max(-y1, min(imh - 1 - y2, shift_y))

        bbox = [x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y]
        return bbox


class ScaleBBox:
    def __init__(self, helper, scale):
        self.helper = helper
        self.scale = scale
        self.scale_x = None
        self.scale_y = None

    def __call__(self, bbox, shape):
        scale_val = (
            1 + (self.helper["random_number"](-1, 1) * self.scale),
            1 + (self.helper["random_number"](-1, 1) * self.scale),
        )
        x1, y1, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        scale_x, scale_y = scale_val[0], scale_val[1]
        # corner form and center form created
        # calculate corner
        # x2 = x1 + w
        # y2 = y1 + h
        # calculate center
        x, y = x1 + w // 2, y1 + h // 2
        imh, imw = shape[:2]
        scale_x = min(scale_x, float(imw) / w)
        scale_y = min(scale_y, float(imh) / h)

        new_x, new_y, new_w, new_h = x, y, w * scale_x, h * scale_y
        self.scale_x, self.scale_y = scale_x, scale_y

        return self.helper["center2corner"]([new_x, new_y, new_w, new_h])
