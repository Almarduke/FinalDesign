"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from . import BASE_OPTION, MODEL_OPTION
from dataloader import find_dataset_class


class Options:
    def __init__(self):
        for (key, value) in BASE_OPTION.items():
            setattr(self, key, value)

        for (key, value) in MODEL_OPTION.items():
            setattr(self, key, value)

        dataset_class = find_dataset_class(self.dataset)
        for (key, value) in dataset_class.option.items():
            setattr(self, key, value)

        self.checkpoints_dir = os.path.join(os.getcwd(), self.checkpoints_dir)
        self.images_dir = os.path.join(os.getcwd(), self.images_dir)
        self.dataset_dir = os.path.join(os.getcwd(), self.dataset_dir)

        # 语义标签，label + 无法识别
        self.n_semantic = self.n_label
        if self.contain_dontcare_label:
            self.n_semantic += 1
