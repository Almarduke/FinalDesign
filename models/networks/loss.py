"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.util import get_float_tensor


# Defines the GAN hinge_loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    # 如果target_is_real,那么input=pred_real,且越大越好
    # 如果target_is_fake,那么input=pred_fake,且越大越好
    # GANLOSS使用hingeloss（比crossentropy好）
    # HingeLoss(y) = max(0, 1−ty)
    # 注意正常情况下input是一个四维tensor，但是如果用了multiscalediscriminator，input就是一个四维tensor的list
    def forward(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                loss = self.hinge_loss(pred_i, target_is_real, for_discriminator)
            return loss / len(input)
        else:
            return hinge_loss(input, target_is_real, for_discriminator)

    def hinge_loss(self, input, target_is_real, for_discriminator):
        if for_discriminator:
            t = 1 if target_is_real else -1
            hinge = torch.max(1 - t * input, self.zero_tensor_of_size(input))
            return torch.mean(hinge)
        else:
            assert target_is_real, "The generator's hinge loss must be aiming for real"
            return -torch.mean(input)

    def zero_tensor_of_size(self, input):
        Tensor = type(input)
        zero_tensor = Tensor(1).fill_(0)
        zero_tensor.requires_grad_(False)
        zero_tensor = zero_tensor.expand_as(input)
        zero_tensor = zero_tensor.cuda() if input.is_cuda else zero_tensor
        return zero_tensor


    # def hinge_loss(self, input, target_is_real, for_discriminator=True):
    #     if for_discriminator:
    #         t = 1 if target_is_real else -1
    #         hinge = torch.max(1 - t * input, self.zero_tensor_of_size(input))
    #         return torch.mean(hinge)
    #     else:
    #         assert target_is_real, "The generator's hinge loss must be aiming for real"
    #         return -torch.mean(input)


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
# class VGGLoss(nn.Module):
#     def __init__(self):
#         super(VGGLoss, self).__init__()
#         self.vgg = VGG19().cuda()
#         self.criterion = nn.L1Loss()
#         self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
#
#     def forward(self, x, y):
#         x_vgg, y_vgg = self.vgg(x), self.vgg(y)
#         loss = 0
#         for i in range(len(x_vgg)):
#             loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
#         return loss


# KL Divergence hinge_loss used in VAE with an image encoder
# VAE和KLD相关：https://toutiao.io/posts/387ohs/preview
class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, mu, logvar):
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
