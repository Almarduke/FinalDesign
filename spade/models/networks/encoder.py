import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from spade.models.networks.base_network import BaseNetwork
from spade.util.train_util import load_network, get_device, use_gpu


class ConvEncoder(BaseNetwork):
    # 新建encoder，必要的话从保存的文件读取
    @staticmethod
    def create_network(opt):
        if not opt.use_vae:
            return None
        netE = ConvEncoder(opt)
        netE = netE.to(get_device(opt)) if use_gpu(opt) else netE
        netE.init_weights(opt.init_variance)
        if not opt.is_train or opt.current_epoch > 0:
            netE = load_network(netE, 'E', opt.current_epoch, opt)
        return netE

    def __init__(self, opt):
        super().__init__()
        # 3*3-⬇2-Conv-64, IN, LReLU
        self.conv64 = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(64, affine=False),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-128, IN, LReLU
        self.conv128 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-256, IN, LReLU
        self.conv256 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-512, IN, LReLU
        self.conv512_0 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(512, affine=False),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-512, IN, LReLU
        self.conv512_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(512, affine=False),
            nn.LeakyReLU(0.2, False)
        )
        # 3*3-⬇2-Conv-512, IN, LReLU
        self.conv512_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(512, affine=False),
            nn.LeakyReLU(0.2, False)
        )

        # 把4*4*512的图像展开后全联接到256维
        # 512是最后一个卷积层的输出channel
        w = h = 4
        self.fc_mu = nn.Linear(512 * w * h, opt.z_dim)
        self.fc_logvar = nn.Linear(512 * w * h, opt.z_dim)
        self.opt = opt

    # https://toutiao.io/posts/387ohs/preview
    # 怎么找出专属于 Xk 的正态分布 p(Z|Xk) 的均值和方差呢？我就用神经网络来拟合出来。
    # 于是我们构建两个神经网络 μk=f1(Xk)，logσ^2=f2(Xk) 来算它们了。
    # 我们选择拟合 logσ^2 而不是直接拟合 σ^2，是因为
    # σ^2 总是非负的，需要加激活函数处理，而拟合 logσ^2 不需要加激活函数，因为它可正可负。
    def forward(self, x):
        x = self.resize(x)
        x = self.conv64(x)
        x = self.conv128(x)
        x = self.conv256(x)
        x = self.conv512_0(x)
        x = self.conv512_1(x)
        x = self.conv512_2(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    # 如果图像不是256 * 256的话，把图像进行缩放
    # 所以上面 w = h = 4 （4 = 256 / (2^6)）
    def resize(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x

