"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.spade_resblk import VGG19


# Defines the GAN hinge_loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.Tensor = tensor
        self.opt = opt

    def zero_tensor_of_size(self, input):
        zero_tensor = self.Tensor(1).zero_()
        zero_tensor.requires_grad_(False)
        return zero_tensor.expand_as(input)

    # 如果target_is_real,那么input=pred_real,且越大越好
    # 如果target_is_fake,那么input=pred_fake,且越大越好
    # HingeLoss(y) = max(0, 1−ty)
    def loss(self, input, target_is_real, for_discriminator=True):
        if for_discriminator:
            t = 1 if target_is_real else -1
            hinge = torch.max(1 - t * input, self.zero_tensor_of_size(input))
            return torch.mean(hinge)
        else:
            assert target_is_real, "The generator's hinge loss must be aiming for real"
            return -torch.mean(input)

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            all_img_loss = sum([self.loss(pred_i, target_is_real, True) for pred_i in input])
            return all_img_loss / len(input)
        else:
            return self.hinge_loss(input, target_is_real)


    # def get_target_tensor(self, input, target_is_real):
    #     if target_is_real:
    #         if self.real_label_tensor is None:
    #             self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
    #             self.real_label_tensor.requires_grad_(False)
    #         return self.real_label_tensor.expand_as(input)
    #     else:
    #         if self.fake_label_tensor is None:
    #             self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
    #             self.fake_label_tensor.requires_grad_(False)
    #         return self.fake_label_tensor.expand_as(input)


# Perceptual hinge_loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence hinge_loss used in VAE with an image encoder
# VAE和KLD相关：https://toutiao.io/posts/387ohs/preview
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
