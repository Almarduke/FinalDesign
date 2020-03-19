"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork


class ConvEncoder(BaseNetwork):
    # 新建encoder，必要的话从保存的文件读取
    @staticmethod
    def create_network(opt):
        netE = ConvEncoder(opt) if opt.use_vae else None
        if opt.continue_train:
            netE = util.load_network(netE, 'E', opt.epoch, opt)
        return netE

    def __init__(self, opt):
        super().__init__()
        # 3*3-⬇2-Conv-64, IN, LReLU
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-128, IN, LReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-256, IN, LReLU
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-512, IN, LReLU
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-512, IN, LReLU
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-512, IN, LReLU
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, False)
        )

        # 把4*4*512的图像展开后全联接到256维
        # 512是最后一个卷积层的输出channel
        w = h = 4
        self.fc_mu = nn.Linear(512 * w * h, 256)
        self.fc_var = nn.Linear(512 * w * h, 256)
        self.opt = opt

    # https://toutiao.io/posts/387ohs/preview
    # 怎么找出专属于 Xk 的正态分布 p(Z|Xk) 的均值和方差呢？我就用神经网络来拟合出来。
    # 于是我们构建两个神经网络 μk=f1(Xk)，logσ^2=f2(Xk) 来算它们了。
    # 我们选择拟合 logσ^2 而不是直接拟合 σ^2，是因为
    # σ^2 总是非负的，需要加激活函数处理，而拟合 logσ^2 不需要加激活函数，因为它可正可负。
    def forward(self, x):
        x = self.resize(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    # 如果图像不是256 * 256的话，把图像进行缩放
    # 所以上面 w = h = 4 （4 = 256 / (2^6)）
    def resize(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        return x

