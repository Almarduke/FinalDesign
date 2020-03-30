"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import random

from PIL import Image
from .base_dataset import BaseDataset, get_transform
from .utils import get_data_paths
from options import NATURAL_DATASET_OPTION


class NaturalDataset(BaseDataset):
    option = NATURAL_DATASET_OPTION

    # 数据集初始化时读取自己对应文件夹下的图片路径
    # 并检查路径是否正确（一张图像对应一张标签）
    def __init__(self, opt):
        super(NaturalDataset, self).__init__()
        label_paths, img_paths = get_data_paths(opt, sort=True)
        self.pairing_check(opt, label_paths, img_paths)
        self.labels = label_paths
        self.imgs = img_paths
        self.opt = opt
        self.dataset_size = len(self.labels)

    def __len__(self):
        return self.dataset_size

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
        label_tensor = label_transform(label)

        # input image (real images)
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_transform = get_transform(self.opt, img_flip, normalize=True)
        img_tensor = img_transform(img)

        return label_tensor, img_tensor

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    # def postprocess(self, label_tensor):
    #     label_tensor[label_tensor == 255] = self.opt.n_label  # 'unknown' is opt.n_label
    #     label_tensor = label_tensor - 1
    #     label_tensor[label_tensor == -1] = self.opt.n_label
    #     return label_tensor
