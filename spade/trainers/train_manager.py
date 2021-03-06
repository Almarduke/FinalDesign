import torch
from spade.util.train_util import use_gpu


# Trainer类负责管理model和optimizer
# 更新网络权重和计算loss
class TrainManager:
    def __init__(self, opt, gan_loss, kld_loss, vgg_loss, feature_loss):
        self.opt = opt
        self.old_lr = opt.learning_rate
        self.gan_loss = gan_loss
        self.kld_loss = kld_loss
        self.vgg_loss = vgg_loss
        self.feature_loss = feature_loss

    # 优化器的创建（仅在train时调用）
    def create_optimizers(self, opt, spade_gan):
        model = spade_gan.module if use_gpu(opt) else spade_gan
        D_params = list(model.netD.parameters())
        G_params = list(model.netG.parameters())
        if opt.use_vae:
            G_params += list(model.netE.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        G_lr = opt.learning_rate / 2 if opt.TTUR else opt.learning_rate
        D_lr = opt.learning_rate * 2 if opt.TTUR else opt.learning_rate
        optG = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optD = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return optG, optD

    # train时更新generator的权重
    def get_lossG(self, seg_maps, real_imgs, spade_gan):
        fake_imgs, mu, logvar, pred_fake, pred_real = spade_gan(seg_maps, real_imgs, for_discriminator=False)
        lossGAN = self.gan_loss(pred_fake, True, False)
        lossKLD = self.kld_loss(mu, logvar) * self.opt.lambda_kld
        lossVGG = self.vgg_loss(fake_imgs, real_imgs) * self.opt.lambda_vgg
        lossFeature = self.feature_loss(pred_fake, pred_real) * self.opt.lambda_feature
        print(f'lossGAN {lossGAN}, lossKLD: {lossKLD}, lossVGG: {lossVGG}, lossFeature: {lossFeature}', flush=True)
        lossG = lossGAN + lossKLD + lossVGG + lossFeature
        return lossG, fake_imgs

    # train时更新discriminator的权重
    def get_lossD(self, seg_maps, real_imgs, spade_gan):
        # fake_imgs, mu, logvar, pred_fake, pred_real = spade_gan(seg_maps, real_imgs)
        pred_fake, pred_real = spade_gan(seg_maps, real_imgs, for_discriminator=True)
        lossFake = self.gan_loss(pred_fake, False, True)
        lossReal = self.gan_loss(pred_real, True, True)
        print(f'lossFake: {lossFake}, lossReal: {lossReal}', flush=True)
        lossD = lossFake + lossReal
        return lossD

    # 更新学习率learning rate decay
    def update_learning_rate(self, epoch, optG, optD):
        if epoch >= self.opt.total_epochs - self.opt.decay_epochs:
            lr_decay = self.opt.learning_rate / self.opt.decay_epochs
            decay_num = epoch - (self.opt.total_epochs - self.opt.decay_epochs)
            new_lr = self.opt.learning_rate - lr_decay * decay_num

            if self.opt.TTUR:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2
            else:
                new_lr_G = new_lr
                new_lr_D = new_lr

            for param_group in optG.param_groups:
                param_group['lr'] = new_lr_G
            for param_group in optD.param_groups:
                param_group['lr'] = new_lr_D
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr), flush=True)
            self.old_lr = new_lr

