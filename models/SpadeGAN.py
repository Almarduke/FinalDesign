"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import util.util as util
from .networks.loss import GANLoss, KLDLoss
from .networks.encoder import ConvEncoder
from .networks.generator import SpadeGenerator
from .networks.discriminator import MultiscaleDiscriminator


class SpadeGAN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.netG = SpadeGenerator.create_network(opt)
        self.netD = MultiscaleDiscriminator.create_network(opt)
        self.netE = ConvEncoder.create_network(opt)

    # 所有的forward都从同一个入口
    # 因为DataParallel不能并行处理自定义方法
    # 添加mode参数来控制
    def forward(self, data):
        seg_map, real_image = data
        if self.opt.is_train:
            fake_image, mu, logvar = self.generate_fake(seg_map, real_image)
            pred_fake, pred_real = self.discriminate(seg_map, fake_image, real_image)
            return fake_image, mu, logvar, pred_fake, pred_real
        else:
            fake_image, mu, logvar = self.generate_fake(seg_map, None)
            return fake_image, mu, logvar

    # =====================================================
    # encoder的编码和重参数化
    # =====================================================
    def encode_z(self, real_image):
        mu, var = self.netE(real_image)
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
    def generate_fake(self, seg_map, real_image):
        z, mu, logvar = None, None, None
        if self.opt.use_vae and real_image:
            z, mu, logvar = self.encode_z(real_image)
        fake_image = self.netG(seg_map, z=z)
        return fake_image, mu, logvar

    # =====================================================
    # 计算Discriminator的相关方法
    # =====================================================

    # 对于每一组fake/real image，输入netD之后返回预测结果
    # 先将fake/real image分别和instance上下重叠，之后将两者左右连接
    # 为了避免BatchNorm的时候两边的统计数据不同
    # 所以把两边一起喂给netD
    def discriminate(self, seg_map, fake_image, real_image):
        # mutilscale GAN是多个gan（coarse to fine）
        # 对于多张图像进行处理
        def divide_pred(pred):
            if type(pred) == list:
                fake, real = [], []
                for tensor in pred:
                    fake.append(tensor[:tensor.size(0) // 2])
                    real.append(tensor[tensor.size(0) // 2:])
            else:
                fake = pred[:pred.size(0) // 2]
                real = pred[pred.size(0) // 2:]
            return fake, real

        fake_concat = torch.cat([seg_map, fake_image], dim=1)
        real_concat = torch.cat([seg_map, real_image], dim=1)
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

