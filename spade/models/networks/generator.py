import torch
import torch.nn as nn
import torch.nn.functional as F
from spade.models.networks.base_network import BaseNetwork
from spade.models.networks.spade_resblk import SpadeResblk
from spade.util.train_util import load_network, get_device, use_gpu


class SpadeGenerator(BaseNetwork):
    # 新建Generator，必要的话从保存的文件读取
    @staticmethod
    def create_network(opt):
        netG = SpadeGenerator(opt)
        netG = netG.to(get_device(opt)) if use_gpu(opt) else netG
        netG.init_weights(opt.init_variance)
        if not opt.is_train or opt.current_epoch > 0:
            netG = load_network(netG, 'G', opt.current_epoch, opt)
        return netG

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        self.fc = nn.Linear(opt.z_dim, 1024 * self.sw * self.sh)

        self.resblk1024_0 = SpadeResblk(1024, 1024, opt)
        self.resblk1024_1 = SpadeResblk(1024, 1024, opt)
        self.resblk1024_2 = SpadeResblk(1024, 1024, opt)

        self.resblk512 = SpadeResblk(1024, 512, opt)
        self.resblk256 = SpadeResblk(512, 256, opt)
        self.resblk128 = SpadeResblk(256, 128, opt)
        self.resblk64 = SpadeResblk(128, 64, opt)

        self.conv_img = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, seg, z=None):
        if z is None:
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=seg.get_device())
        x = self.fc(z)
        x = x.view(-1, 1024, self.sh, self.sw)

        x = self.upsample(self.resblk1024_0(x, seg))
        x = self.upsample(self.resblk1024_1(x, seg))
        x = self.upsample(self.resblk1024_2(x, seg))
        x = self.upsample(self.resblk512(x, seg))
        x = self.upsample(self.resblk256(x, seg))
        x = self.upsample(self.resblk128(x, seg))

        x = self.resblk64(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

    def compute_latent_vector_size(self, opt):
        num_up_layers = 6
        sw = opt.load_size[0] // (2 ** num_up_layers)
        sh = opt.load_size[1] // (2 ** num_up_layers)
        return sw, sh
