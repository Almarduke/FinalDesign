"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from util.util import load_network
from models.networks.base_network import BaseNetwork


class MultiscaleDiscriminator(BaseNetwork):
    # 新建Discriminator，必要的话从保存的文件读取
    @staticmethod
    def create_network(opt):
        if not opt.is_train:
            return None
        netD = MultiscaleDiscriminator(opt)
        netD = netD.cuda() if torch.cuda.is_available() else netD
        netD.init_weights(opt.init_variance)
        if not opt.is_train or opt.continue_train:
            netD = load_network(netD, 'D', opt.current_epoch, opt)
        return netD

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        for i in range(opt.D_model_num):
            subnetD = NLayerDiscriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        # return F.avg_pool2d(input, kernel_size=2, stride=2)
        w, h = input.size()[2:]
        w = int(w // 2)
        h = int(h // 2)
        return F.interpolate(input, size=(w, h), mode='nearest')

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.D_model_num x 4 (4 is layer num of D)
    def forward(self, input_image):
        result = []
        for name, D in self.named_children():
            out = D(input_image)
            result.append(out)
            input_image = self.downsample(input_image)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        input_nc = self.compute_D_input_nc(opt)

        # 4*4-⬇2-Conv-64, LReLU
        self.conv64 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, False)
        )
        # 4*4-⬇2-Conv-128, IN, LReLU
        self.conv128 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2)),
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, False)
        )
        # 4*4-⬇2-Conv-256, IN, LReLU
        self.conv256 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2)),
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, False)
        )
        # 4*4-Conv-512, IN, LReLU
        self.conv512 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=2)),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, False)
        )
        # 输出识别结果
        self.conv_out = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=2)

    def forward(self, input_image):
        out_layer1 = self.conv64(input_image)
        out_layer2 = self.conv128(out_layer1)
        out_layer3 = self.conv256(out_layer2)
        out_layer4 = self.conv512(out_layer3)
        out_final = self.conv_out(out_layer4)
        return [out_layer1, out_layer2, out_layer3, out_layer4, out_final]

    # 注意segmap做过onehot了，并且包含id=0（dont care label）
    # 所以总共是 RGB(3) + input_nc(150) + dontcare(1)
    def compute_D_input_nc(self, opt):
        input_nc = opt.n_semantic + opt.output_nc
        return input_nc
