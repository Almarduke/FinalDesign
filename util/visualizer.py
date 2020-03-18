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
            self.log_name = os.path.join(opt.checkpoints_dir, opt.dataset, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime('%c')
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    # def display_current_results(self, visuals, epoch, step):
    #
    #     ## convert tensors to numpy arrays
    #     visuals = self.convert_visuals_to_numpy(visuals)
    #
    #     if self.tf_log: # show images in tensorboard output
    #         img_summaries = []
    #         for label, image_numpy in visuals.items():
    #             # Write the image to a string
    #             try:
    #                 s = StringIO()
    #             except:
    #                 s = BytesIO()
    #             if len(image_numpy.shape) >= 4:
    #                 image_numpy = image_numpy[0]
    #             scipy.misc.toimage(image_numpy).save(s, format="jpeg")
    #             # Create an Image object
    #             img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
    #             # Create a Summary value
    #             img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))
    #
    #         # Create and write Summary
    #         summary = self.tf.Summary(value=img_summaries)
    #         self.writer.add_summary(summary, step)

    # # errors: dictionary of error labels and values
    # def plot_current_errors(self, errors, step):
    #     if self.tf_log:
    #         for tag, value in errors.items():
    #             value = value.mean().float()
    #             summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
    #             self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, batch_id, running_time, loss):
        message = '(epoch: %d, iters: %d, time: %.2f sec, loss: %.3f) ' % (epoch, batch_id, running_time, loss.item())
        print(message)
        if self.opt.save_log:
            with open(self.log_name, 'a') as log_file:
                log_file.write('%s\n' % message)

    # def convert_visuals_to_numpy(self, visuals):
    #     for key, t in visuals.items():
    #         tile = self.opt.batchSize > 8
    #         if 'input_label' == key:
    #             t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
    #         else:
    #             t = util.tensor2img(t, tile=tile)
    #         visuals[key] = t
    #     return visuals

    # save image to the disk
    # def save_images(self, webpage, visuals, image_path):
    #     visuals = self.convert_visuals_to_numpy(visuals)
    #
    #     image_dir = webpage.get_image_dir()
    #     short_path = ntpath.basename(image_path[0])
    #     name = os.path.splitext(short_path)[0]
    #
    #     webpage.add_header(name)
    #     ims = []
    #     txts = []
    #     links = []
    #
    #     for label, image_numpy in visuals.items():
    #         image_name = os.path.join(label, '%s.png' % (name))
    #         save_path = os.path.join(image_dir, image_name)
    #         util.save_image(image_numpy, save_path, create_dir=True)
    #
    #         ims.append(image_name)
    #         txts.append(label)
    #         links.append(image_name)
    #     webpage.add_images(ims, txts, links, width=self.win_size)
