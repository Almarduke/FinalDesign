"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
from . import util
import scipy.misc
import tensorflow as tf


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        if opt.is_train and opt.save_log:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.dataset)
            self.log_name = os.path.join(opt.checkpoints_dir, opt.dataset, 'loss_log.txt')
            util.mkdir(self.log_dir)
            with open(self.log_name, "a") as log_file:
                now = time.strftime('%c')
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, batch_id, running_time, loss):
        message = '(epoch: %d, iters: %d, time: %.2f sec, hinge_loss: %.3f) ' % (epoch, batch_id, running_time, loss.item())
        print(message, flush=True)
        if self.opt.save_log:
            with open(self.log_name, 'a') as log_file:
                log_file.write('%s\n' % message)

    # def convert_visuals_to_numpy(self, visuals):
    #     for key, t in visuals.items():
    #         tile = self.opt.batchSize > 8
    #         if 'input_label' == key:
    #             t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
    #         else:
    #             t = util.tensor2im(t, tile=tile)
    #         visuals[key] = t
    #     return visuals

    def save_images(self, epoch, iter, labels, real_imgs, generated_imgs):
        img_dir = os.path.join(opt.images_dir, opt.dataset)
        for i in range(labels.size()[0]):
            img_name = f'e{epoch}-i{iter}-{i}.png'
            label_img = tensor2label(labels[i], opt)
            real_img = tensor2img(real_imgs[i], opt)
            generated_img = tensor2img(generated_imgs[i], opt)

            label_dir = os.path.join(img_dir, 'label')
            real_dir = os.path.join(img_dir, 'real')
            generate_dir = os.path.join(img_dir, 'generate')

            util.save_image(label_img, label_dir, img_name)
            util.save_image(real_img, real_dir, img_name)
            util.save_image(generated_img, generate_dir, img_name)

