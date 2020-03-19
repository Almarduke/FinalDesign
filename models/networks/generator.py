"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.spade_resblk import SpadeResblk


class SpadeGenerator(BaseNetwork):
    # 新建Generator，必要的话从保存的文件读取
    @staticmethod
    def create_network(opt):
        netG = SpadeGenerator(opt)
        if not opt.is_train or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.epoch, opt)
        return netG

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # In case of VAE, we will sample from random z vector
        # Otherwise, we make the network deterministic by starting with
        # downsampled segmentation map instead of random z
        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 1024 * self.sw * self.sh)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 1024, 3, padding=1)

        self.resblk1024 = SpadeResblk(1024, 1024, opt)
        self.resblk512 = SpadeResblk(1024, 512, opt)
        self.resblk256 = SpadeResblk(512, 256, opt)
        self.resblk128 = SpadeResblk(256, 128, opt)
        self.resblk64 = SpadeResblk(128, 64, opt)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_img = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, seg, z=None):
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=seg.get_device())
            x = self.fc(z)
            x = x.view(-1, 1024, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.upsample(self.resblk1024(x, seg))
        x = self.upsample(self.resblk1024(x, seg))
        x = self.upsample(self.resblk1024(x, seg))
        x = self.upsample(self.resblk512(x, seg))
        x = self.upsample(self.resblk256(x, seg))
        x = self.upsample(self.resblk128(x, seg))
        x = self.resblk64(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

    def compute_latent_vector_size(self, opt):
        num_up_layers = 6
        sw = opt.load_size[0] // (2 ** num_up_layers)
        sh = opt.load_size[1] // (2 ** num_up_layers)
        return sw, sh
