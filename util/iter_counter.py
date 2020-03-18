"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import time
import numpy as np
import tqdm


# Helper class that keeps track of training iterations
class EpochCounter:
    def __init__(self, opt):
        self.opt = opt
        self.total_epochs = opt.total_epochs
        self.current_epoch = 0
        self.epoch_start_time = time.time()
        self.record_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset, 'iter.txt')

        if opt.is_train and opt.continue_train:
            try:
                self.current_epoch = int(np.loadtxt(self.record_path, dtype=int))
                print(f'Dataset {self.opt.dataset} Resuming from epoch {self.current_epoch}')
            except:
                print(f'Could not load iteration record at {self.record_path}. Starting from beginning.')

    # return the iterator of epochs for the training
    def training_epochs(self):
        return tqdm.trange(self.current_epoch, self.total_epochs)

    def record_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        self.current_epoch = epoch

    def record_epoch_end(self):
        current_time = time.time()
        epoch_running_time = current_time - self.epoch_start_time
        np.savetxt(self.record_path, self.current_epoch + 1, fmt='%d')
        print(f'End of epoch {self.current_epoch} / {self.total_epochs} \t Time Taken: {epoch_running_time} sec')
        print(f'Current epoch count saved at {self.record_path}')