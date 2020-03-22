"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils import spectral_norm
from models.networks.normalization import SPADE


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SpadeResblk(nn.Module):
    def __init__(self, fin, fout, opt):
        super(SpadeResblk, self).__init__()
        fmiddle = min(fin, fout)
        self.learned_shortcut = (fin != fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        self.spade_0 = SPADE(fin, opt.n_semantic)
        self.spade_1 = SPADE(fmiddle, opt.n_semantic)
        self.spade_s = SPADE(fin, opt.n_semantic)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.spade_0(x, seg)))
        dx = self.conv_1(self.actvn(self.spade_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        x_s = self.conv_s(self.spade_s(x, seg)) if self.learned_shortcut else x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
# class ResnetBlock(nn.Module):
#     def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
#         super().__init__()
#
#         pw = (kernel_size - 1) // 2
#         self.conv_block = nn.Sequential(
#             nn.ReflectionPad2d(pw),
#             norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
#             activation,
#             nn.ReflectionPad2d(pw),
#             norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
#         )
#
#     def forward(self, x):
#         y = self.conv_block(x)
#         out = x + y
#         return out


# VGG architecter, used for the perceptual hinge_loss using a pretrained VGG network
# class VGG19(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super().__init__()
#         vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(2):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
#
#     def forward(self, X):
#         h_relu1 = self.slice1(X)
#         h_relu2 = self.slice2(h_relu1)
#         h_relu3 = self.slice3(h_relu2)
#         h_relu4 = self.slice4(h_relu3)
#         h_relu5 = self.slice5(h_relu4)
#         out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#         return out
