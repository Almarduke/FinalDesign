"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from . import BASE_OPTION, MODEL_OPTION
from dataloader import find_dataset_class


class Options:
    def __init__(self):
        for (key, value) in BASE_OPTION:
            setattr(self, key, value)

        for (key, value) in MODEL_OPTION:
            setattr(self, key, value)

        dataset_class = find_dataset_class(self.dataset)
        for (key, value) in dataset_class.option:
            setattr(self, key, value)
