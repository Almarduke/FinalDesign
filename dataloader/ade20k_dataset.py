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
            for label_path, img_path in zip(label_paths, img_paths):
                label_name = os.path.splitext(os.path.basename(label_path))[0]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                assert label_name == img_name, \
                    "请检查数据集，图像和标签的文件名不匹配"
                assert label_path.endswith('png') and img_path.endswith('jpg'), \
                    "标签文件必须是png格式, 图像文件必须是jpg格式"

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
        label_tensor = label_transform(label) * 255.0

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
