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

    def pairing_check(self, opt, label_paths, img_paths):
        if opt.pairing_check:
            for label_path, img_path in zip(label_paths, img_paths):
                label_name = os.path.splitext(os.path.basename(label_path))[0]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                assert label_name == img_name, \
                    "请检查数据集，图像和标签的文件名不匹配"
                assert label_path.endswith('png') and img_path.endswith('jpg'), \
                    "标签文件必须是png格式, 图像文件必须是jpg格式"


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
        three_channel_means = (opt.img_mean, opt.img_mean, opt.img_mean)
        three_channel_vars = (opt.img_var, opt.img_var, opt.img_var)
        transform_list += [transforms.Normalize(three_channel_means, three_channel_vars)]
    return transforms.Compose(transform_list)


def scale_and_crop(img, target_size, method=Image.BICUBIC):
    ow, oh = img.size       # original width / height
    tw, th = target_size    # target width / height
    scale_ratio = max(tw / ow, th / oh)
    sw, sh = int(ow * scale_ratio), int(oh * scale_ratio)  # scaled width / height
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
