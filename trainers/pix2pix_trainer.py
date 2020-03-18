"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel


# Trainer类负责管理model和optimizer
# 更新网络权重和计算loss
class Pix2PixTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model, device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.is_train:
            self.optimizer_G, self.optimizer_D = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.learning_rate

    # train时更新generator的权重
    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    # train时更新discriminator的权重
    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    # 更新学习率learning rate decay
    def update_learning_rate(self, epoch):
        if epoch > self.opt.total_epochs - self.opt.decay_epochs:
            lr_decay = self.opt.learning_rate / self.opt.decay_epochs
            new_lr = self.old_lr - lr_decay

            if self.opt.TTUR:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2
            else:
                new_lr_G = new_lr
                new_lr_D = new_lr

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)
