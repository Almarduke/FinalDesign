import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from spade.models.networks.base_network import BaseNetwork
from spade.util.train_util import load_network, get_device, use_gpu


class MultiscaleDiscriminator(BaseNetwork):
    # 新建Discriminator，必要的话从保存的文件读取
    @staticmethod
    def create_network(opt):
        if not opt.is_train:
            return None
        netD = MultiscaleDiscriminator(opt)
        netD = netD.to(get_device(opt)) if use_gpu(opt) else netD
        netD.init_weights(opt.init_variance)
        if not opt.is_train or opt.current_epoch > 0:
            netD = load_network(netD, 'D', opt.current_epoch, opt)
        return netD

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        for i in range(opt.D_model_num):
            subnetD = NLayerDiscriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        # return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
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
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, False)
        )
        # # 4*4-Conv-512, IN, LReLU
        self.conv512 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=2)),
            nn.InstanceNorm2d(512, affine=False),
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

    # def replica_network(self):
    #     old_names = ['model0', 'model1', 'model2', 'model3']
    #     new_names = ['resblk_1024_0', 'resblk_1024_1', 'resblk_1024_2', 'resblk_512', 'resblk_256', 'resblk_128', 'resblk_64']
    #     for (old_name, new_name) in zip(old_names, new_names):
    #         layer = getattr(self, old_name)
    #         layer.norm_0.shared = layer.norm_0.mlp_shared
    #         layer.norm_0.gamma = layer.norm_0.mlp_gamma
    #         layer.norm_0.beta = layer.norm_0.mlp_beta
    #         layer.norm_1.shared = layer.norm_1.mlp_shared
    #         layer.norm_1.gamma = layer.norm_1.mlp_gamma
    #         layer.norm_1.beta = layer.norm_1.mlp_beta
    #         if hasattr(layer, 'norm_s'):
    #             layer.norm_s.shared = layer.norm_s.mlp_shared
    #             layer.norm_s.gamma = layer.norm_s.mlp_gamma
    #             layer.norm_s.beta = layer.norm_s.mlp_beta
    #         setattr(self, new_name, layer)
    #         delattr(self, old_name)
    #     self.fc = nn.Conv2d(self.opt.n_semantic, 1024, 3, padding=1)
