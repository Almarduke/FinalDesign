import os
import ntpath
import time
from .train_util import mkdir, tensor2img, tensor2colorlabel, save_image
import scipy.misc
import tensorflow as tf


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        if opt.is_train and opt.save_log:
            self.log_dir = mkdir(os.path.join(opt.checkpoints_dir, opt.dataset))
            self.log_name = os.path.join(opt.checkpoints_dir, opt.dataset, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime('%c')
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def print_current_errors(self, epoch, batch_id, running_time, lossG, lossD):
        message = '(epoch: %d, iters: %d, time: %.2f sec, lossG: %.3f, lossD: %.3f) ' \
                  % (epoch, batch_id, running_time, lossG, lossD)
        print(message, flush=True)
        if self.opt.save_log:
            with open(self.log_name, 'a') as log_file:
                log_file.write('%s\n' % message)

    def save_train_images(self, epoch, iter, label_images, real_images, generated_images):
        image_dir = mkdir(os.path.join(self.opt.images_dir, self.opt.dataset))
        for i in range(label_images.size()[0]):
            img_name = f'e{epoch}-i{iter}-{i}.png'
            label_img = tensor2colorlabel(label_images[i], self.opt)
            real_img = tensor2img(real_images[i], self.opt)
            generated_img = tensor2img(generated_images[i], self.opt)

            label_dir = mkdir(os.path.join(image_dir, 'label'))
            real_dir = mkdir(os.path.join(image_dir, 'real'))
            generate_dir = mkdir(os.path.join(image_dir, 'spade_generate'))

            save_image(label_img, label_dir, img_name)
            save_image(real_img, real_dir, img_name)
            save_image(generated_img, generate_dir, img_name)
        print(f'Epoch {epoch}, iter {iter}, generated images saved.', flush=True)

