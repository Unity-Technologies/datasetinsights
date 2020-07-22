import random

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
