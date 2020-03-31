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
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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


