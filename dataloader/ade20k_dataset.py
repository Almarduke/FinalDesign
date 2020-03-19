"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import random

from PIL import Image
from .base_dataset import BaseDataset, get_transform
from .utils import get_paths
from options import ADE20K_DATASET_OPTION


class ADE20KDataset(BaseDataset):
    option = ADE20K_DATASET_OPTION

    # 数据集初始化时读取自己对应文件夹下的图片路径
    # 并检查路径是否正确（一张图像对应一张标签）
    def __init__(self, opt):
        super(ADE20KDataset, self).__init__()
        label_paths, img_paths = get_paths(opt, sort=True)
        if opt.pairing_check:
            for img_path, label_path in zip(label_paths, img_paths):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                label_name = os.path.splitext(os.path.basename(label_path))[0]

                assert img_name == label_name, \
                    "The label-image pair seems to be wrong since file names are different."

        self.imgs = img_paths
        self.labels = label_paths
        self.dataset_size = len(self.labels)
        self.opt = opt

    def __getitem__(self, index):
        # 训练集中的图像有一半概率翻转，数据增强
        # 因为要控制图像和标签一起反转，所以不能把flip放到get_transform里面
        img_flip = self.opt.is_train and self.opt.flip and random.random() > 0.5

        # Label Image
        label_path = self.labels[index]
        label = Image.open(label_path)
        transform_label = get_transform(self.opt, img_flip, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        label_tensor = self.postprocess(label_tensor)

        # input image (real images)
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_transform = get_transform(self.opt, img_flip, normalize=True)
        img_tensor = img_transform(img)

        return img_tensor, label_tensor

    def __len__(self):
        return self.dataset_size

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    def postprocess(self, label_tensor):
        label_tensor = label_tensor - 1
        label_tensor[label_tensor == -1] = self.opt.label_nc
        return label_tensor
