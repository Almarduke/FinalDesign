"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from sync_batchnorm import DataParallelWithCallback
from models.SpadeGAN import SpadeGAN
from models.networks.loss import GANLoss, KLDLoss


# Trainer类负责管理model和optimizer
# 更新网络权重和计算loss
class TrainManager:
    def __init__(self, opt):
        self.opt = opt
        self.old_lr = opt.learning_rate

    # 优化器的创建（仅在train时调用）
    def create_optimizers(self, opt, spade_gan):
        D_params = list(spade_gan.module.netD.parameters())
        G_params = list(spade_gan.module.netG.parameters())
        if opt.use_vae:
            G_params += list(spade_gan.module.netE.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        G_lr = opt.learning_rate / 2 if opt.TTUR else opt.learning_rate
        D_lr = opt.learning_rate * 2 if opt.TTUR else opt.learning_rate
        optG = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optD = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return optG, optD

    # train时更新generator的权重
    def get_lossG(self, seg_maps, real_imgs, spade_gan, gan_loss, kld_loss):
        fake_imgs, mu, logvar, pred_fake, pred_real = spade_gan(seg_maps, real_imgs)
        lossKLD = kld_loss(mu, logvar) * self.opt.lambda_kld
        lossGAN = gan_loss(pred_fake, True, False)
        lossG = (lossKLD * self.opt.lambda_kld + lossGAN).mean()
        return lossG, fake_imgs

    # train时更新discriminator的权重
    def get_lossD(self, seg_maps, real_imgs, spade_gan, gan_loss):
        fake_imgs, mu, logvar, pred_fake, pred_real = spade_gan(seg_maps, real_imgs)
        lossFake = gan_loss(pred_fake, False, True)
        lossReal = gan_loss(pred_real, True, True)
        lossD = (lossFake + lossReal).mean()
        return lossD

    # 更新学习率learning rate decay
    def update_learning_rate(self, epoch, optG, optD):
        if epoch > self.opt.total_epochs - self.opt.decay_epochs:
            lr_decay = self.opt.learning_rate / self.opt.decay_epochs
            new_lr = self.old_lr - lr_decay

            if self.opt.TTUR:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2
            else:
                new_lr_G = new_lr
                new_lr_D = new_lr

            for param_group in optG.param_groups:
                param_group['lr'] = new_lr_G
            for param_group in optD.param_groups:
                param_group['lr'] = new_lr_D
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr), flush=True)
            self.old_lr = new_lr
