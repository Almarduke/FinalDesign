"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass


def get_transform(opt, img_flip, method=Image.BICUBIC, to_tensor=True, normalize=True):

    transform_list = []
    if opt.preprocess_mode == 'resize':
        transform_list.append(transforms.Resize(opt.load_size, interpolation=method))
    elif opt.preprocess_mode == 'scale_and_crop':
        transform_list.append(transforms.Lambda(lambda img: scale_and_crop(img, opt.load_size, method)))

    if opt.is_train and img_flip:
        transform_list.append(transforms.Lambda(lambda img: flip(img)))
    if to_tensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def scale_and_crop(img, target_size, method=Image.BICUBIC):
    ow, oh = img.size       # original width / height
    tw, th = target_size    # target width / height
    scale_ratio = max(tw / ow, th / oh)
    sw, sh = ow * scale_ratio, oh * scale_ratio  # scaled width / height
    scaled_img = img.resize((sw, sh), method)
    return crop(scaled_img, target_size)


def crop(img, crop_size):
    ow, oh = img.size
    w, h = crop_size
    x = random.randint(0, np.maximum(0, ow - w))
    y = random.randint(0, np.maximum(0, oh - h))
    return img.crop((x, y, x + w, y + h))


def flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
