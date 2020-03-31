"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
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

    def __getitem__(self, index):
        # 训练集中的图像有一半概率翻转，数据增强
        # 因为要控制图像和标签一起反转，所以不能把flip放到get_transform里面
        img_flip = self.opt.is_train and self.opt.flip and random.random() > 0.5

        # Label Image
        # 如果有150个label，那么对应的id为1-150
        # id=0表示unknown
        label_path = self.labels[index]
        label = Image.open(label_path)
        label_transform = get_transform(self.opt, img_flip, method=Image.NEAREST, normalize=False)
        label_tensor = label_transform(label) * 255.0

        # input image (real images)
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_transform = get_transform(self.opt, img_flip, normalize=True)
        img_tensor = img_transform(img)

        return label_tensor, img_tensor


def get_transform(opt, img_flip, method=Image.NEAREST, to_tensor=True, normalize=True):
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
        # https://blog.csdn.net/xys430381_1/article/details/85724668?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
        # totensor方法会自动的把[0, 255] -> [0, 1]
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
