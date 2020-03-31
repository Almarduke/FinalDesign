"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import util.util as util
from .networks.loss import GANLoss, KLDLoss
from .networks.encoder import ConvEncoder
from .networks.generator import SpadeGenerator
from .networks.discriminator import MultiscaleDiscriminator


class SpadeGAN(nn.Module):
    def __init__(self, opt):
        super(SpadeGAN, self).__init__()
        self.opt = opt
        self.netG = SpadeGenerator.create_network(opt)
        self.netD = MultiscaleDiscriminator.create_network(opt)
        self.netE = ConvEncoder.create_network(opt)

    # 所有的forward都从同一个入口
    # 因为DataParallel不能并行处理自定义方法
    # 添加mode参数来控制
    def forward(self, seg_maps, real_imgs):
        if self.opt.is_train:
            fake_imgs, mu, logvar = self.generate_fake(seg_maps, real_imgs)
            pred_fake, pred_real = self.discriminate(seg_maps, fake_imgs, real_imgs)
            return fake_imgs, mu, logvar, pred_fake, pred_real
        else:
            fake_imgs, mu, logvar = self.generate_fake(seg_maps, None)
            return fake_imgs, mu, logvar

    # =====================================================
    # encoder的编码和重参数化
    # =====================================================
    def encode_z(self, real_imgs):
        mu, var = self.netE(real_imgs)
        z = self.reparameterize(mu, var)
        return z, mu, var

    # https://toutiao.io/posts/387ohs/preview
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return epsilon * std + mu

    # =====================================================
    # 计算Generator的相关方法
    # =====================================================

    # 使用generator生成图像
    def generate_fake(self, seg_maps, real_imgs):
        z, mu, logvar = None, None, None
        if self.opt.use_vae and real_imgs is not None:
            z, mu, logvar = self.encode_z(real_imgs)
        fake_imgs = self.netG(seg_maps, z=z)
        return fake_imgs, mu, logvar

    # =====================================================
    # 计算Discriminator的相关方法
    # =====================================================

    # 对于每一组fake/real image，输入netD之后返回预测结果
    # 先将fake/real image分别和instance上下重叠，之后将两者左右连接
    # 为了避免BatchNorm的时候两边的统计数据不同
    # 所以把两边一起喂给netD
    def discriminate(self, seg_maps, fake_imgs, real_imgs):
        # mutilscale GAN是多个gan（coarse to fine）
        # 对于多张图像进行处理
        def divide_pred(pred):
            fake, real = [], []
            for one_D_Pred in pred:
                fake.append([layer[:layer.size(0) // 2] for layer in one_D_Pred])
                real.append([layer[layer.size(0) // 2:] for layer in one_D_Pred])
            return fake, real

        fake_concat = torch.cat([seg_maps, fake_imgs], dim=1)
        real_concat = torch.cat([seg_maps, real_imgs], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = divide_pred(discriminator_out)
        return pred_fake, pred_real

    # =====================================================
    # 工具方法：优化器创建、模型保存、数据预处理、是否使用gpu
    # =====================================================

    # # 优化器的创建（仅在train时调用）
    # def create_optimizers(self, opt):
    #     D_params = list(self.netD.parameters())
    #     G_params = list(self.netG.parameters())
    #     if opt.use_vae:
    #         G_params += list(self.netE.parameters())
    #     beta1, beta2 = opt.beta1, opt.beta2
    #     G_lr = opt.learning_rate
    #     D_lr = opt.learning_rate
    #     optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
    #     optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
    #     return optimizer_G, optimizer_D

    # 保存某个epoch时的模型
    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)
        print(f'Model of epoch {epoch} saved', flush=True)

