"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import argparse
import os
import torch
import pickle
from . import BASE_OPTION
from dataloader import find_dataset_class


class Options():
    def __init__(self):
        for (key, value) in BASE_OPTION:
            setattr(self, key, value)

        dataset_class = find_dataset_class(self.dataset)
        for (key, value) in dataset_class.option:
            setattr(self, key, value)

        # model_class = find_model_class(self.model)
        # for (key, value) in dataset_class.option:
        #     setattr(self, key, value)



    # def parse(self, save=False):
    #
    #     opt = self.gather_options()
    #     opt.isTrain = self.isTrain   # train or test
    #
    #     self.print_options(opt)
    #     if opt.isTrain:
    #         self.save_options(opt)
    #
    #     # Set semantic_nc based on the option.
    #     # This will be convenient in many places
    #     opt.semantic_nc = opt.label_nc + \
    #         (1 if opt.contain_dontcare_label else 0) + \
    #         (0 if opt.no_instance else 1)
    #
    #     # set gpu ids
    #     str_ids = opt.gpu_ids.split(',')
    #     opt.gpu_ids = []
    #     for str_id in str_ids:
    #         id = int(str_id)
    #         if id >= 0:
    #             opt.gpu_ids.append(id)
    #     if len(opt.gpu_ids) > 0:
    #         torch.cuda.set_device(opt.gpu_ids[0])
    #
    #     assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
    #         "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
    #         % (opt.batchSize, len(opt.gpu_ids))
    #
    #     self.opt = opt
    #     return self.opt
