"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import SpadeResblk


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

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

    def compute_latent_vector_size(self, opt):
        num_up_layers = 6
        sw = opt.load_size[0] // (2 ** num_up_layers)
        sh = opt.load_size[1] // (2 ** num_up_layers)
        return sw, sh

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

# 这个是Pix2Pix的生成器，不用管了
# class Pix2PixHDGenerator(BaseNetwork):
#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
#         parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
#         parser.add_argument('--resnet_kernel_size', type=int, default=3,
#                             help='kernel size of the resnet block')
#         parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
#                             help='kernel size of the first convolution')
#         parser.set_defaults(norm_G='instance')
#         return parser
#
#     def __init__(self, opt):
#         super().__init__()
#         input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
#
#         norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
#         activation = nn.ReLU(False)
#
#         model = []
#
#         # initial conv
#         model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
#                   norm_layer(nn.Conv2d(input_nc, opt.ngf,
#                                        kernel_size=opt.resnet_initial_kernel_size,
#                                        padding=0)),
#                   activation]
#
#         # downsample
#         mult = 1
#         for i in range(opt.resnet_n_downsample):
#             model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
#                                            kernel_size=3, stride=2, padding=1)),
#                       activation]
#             mult *= 2
#
#         # resnet blocks
#         for i in range(opt.resnet_n_blocks):
#             model += [ResnetBlock(opt.ngf * mult,
#                                   norm_layer=norm_layer,
#                                   activation=activation,
#                                   kernel_size=opt.resnet_kernel_size)]
#
#         # upsample
#         for i in range(opt.resnet_n_downsample):
#             nc_in = int(opt.ngf * mult)
#             nc_out = int((opt.ngf * mult) / 2)
#             model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
#                                                     kernel_size=3, stride=2,
#                                                     padding=1, output_padding=1)),
#                       activation]
#             mult = mult // 2
#
#         # final output conv
#         model += [nn.ReflectionPad2d(3),
#                   nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
#                   nn.Tanh()]
#
#         self.model = *model)
#
#     def forward(self, input, z=None):
#         return self.model(input)
