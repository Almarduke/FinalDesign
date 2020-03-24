"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
from .vgg19 import VGG19


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


# Perceptual hinge_loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_id):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda(gpu_id)
        self.gpu_id = gpu_id
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    # 这个在GPU之间移动的代码是因为fake_imgs这种生成的数据保存在gpu0上
    # 而VGGLoss在gpu3上，所以移动到gpu3来处理，之后再移动回去
    # KLDLoss和GANLoss并没有类似的操作，因为它们虽然是nn.Module
    # 但是本质上只是一个"计算"的封装，自身没有参数，
    # 也就是说kld_loss.cuda(2)之类的代码推测没什么用，还是在处理GPU0上的数据
    def forward(self, fake_imgs, real_imgs):
        fake_imgs = fake_imgs.cuda(self.gpu_id)
        real_imgs = real_imgs.cuda(self.gpu_id)
        fake_vgg, real_vgg = self.vgg(fake_imgs), self.vgg(real_imgs)
        vgg_loss = 0
        for i in range(len(fake_vgg)):
            vgg_loss += self.weights[i] * self.criterion(fake_vgg[i], real_vgg[i].detach())
        return vgg_loss.cuda()


# KL Divergence hinge_loss used in VAE with an image encoder
# VAE和KLD相关：https://toutiao.io/posts/387ohs/preview
class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, mu, logvar):
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
