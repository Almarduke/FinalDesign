import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from spade.models.networks.spade import SPADE


class SpadeResblk(nn.Module):
    def __init__(self, fin, fout, opt):
        super(SpadeResblk, self).__init__()
        fmiddle = min(fin, fout)
        self.learned_shortcut = (fin != fout)
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))

        self.norm_0 = SPADE(fin, opt.n_semantic)
        self.norm_1 = SPADE(fmiddle, opt.n_semantic)

        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))
            self.norm_s = SPADE(fin, opt.n_semantic)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        x_s = self.conv_s(self.norm_s(x, seg)) if self.learned_shortcut else x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
