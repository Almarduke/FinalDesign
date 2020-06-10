import torch.nn as nn
import torch.nn.functional as F
from spade.sync_batchnorm import SynchronizedBatchNorm2d

class SPADE(nn.Module):
    def __init__(self, norm_nc, n_semantic):
        super(SPADE, self).__init__()
        ks, hidden_channel = 3, 128
        pw = ks // 2  # padding width
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        self.shared = nn.Sequential(nn.Conv2d(n_semantic, hidden_channel, kernel_size=ks, padding=pw), nn.ReLU())
        self.gamma = nn.Conv2d(hidden_channel, norm_nc, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(hidden_channel, norm_nc, kernel_size=ks, padding=pw)

    # 首先将输入参数归一化，并将segmap resize到同一大小
    # 用最近邻插值的方法resize图像，质量一般
    # 之后根据segmap提取scale（gamma）和bias（beta），反归一化
    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.shared(segmap)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
