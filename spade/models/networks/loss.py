import torch
import torch.nn as nn
import torchvision


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg19_pretrained = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = vgg19_pretrained[:2]
        self.slice2 = vgg19_pretrained[2:7]
        self.slice3 = vgg19_pretrained[7:12]
        self.slice4 = vgg19_pretrained[12:21]
        self.slice5 = vgg19_pretrained[21:30]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    # 如果target_is_real,那么input=pred_real,且越大越好
    # 如果target_is_fake,那么input=pred_fake,且越小越好
    # GANLOSS使用hingeloss（比crossentropy好）
    # HingeLoss(y) = max(0, 1−ty)
    #
    # 用了multiscalediscriminator，input就是一个四维tensor的list的list，input中的每一项对应一个discriminator的结果
    # 一个discriminator的结果是一个list，list中的每个tensor对应discriminator的每一层的输出
    # gan_loss不计算feature，所以只用最后一项
    # len(pred)是multiscale discriminator中discriminator的数量
    def forward(self, pred, target_is_real, for_discriminator=True):
        loss = 0
        for pred_i in pred:
            pred_i = pred_i[-1] if isinstance(pred_i, list) else pred_i
            if for_discriminator:
                new_loss = self.hinge_loss(pred_i, target_is_real)
            else:
                new_loss = self.generator_loss(pred_i)
            loss += new_loss
        return loss / len(pred)

    def generator_loss(self, pred):
        return -torch.mean(pred)

    def hinge_loss(self, input, target_is_real):
        t = 1 if target_is_real else -1
        hinge = torch.max(1 - t * input, self.zero_tensor_of_size(input))
        return torch.mean(hinge)

    def zero_tensor_of_size(self, input):
        Tensor = type(input)
        zero_tensor = Tensor(1).fill_(0)
        zero_tensor.requires_grad_(False)
        zero_tensor = zero_tensor.expand_as(input)
        zero_tensor = zero_tensor.cuda() if input.is_cuda else zero_tensor
        return zero_tensor


# Perceptual hinge_loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.gpu = None
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def move_to(self, gpu):
        self.vgg = self.vgg.to(gpu)
        self.gpu = gpu

    # 这个在GPU之间移动的代码是因为fake_imgs这种生成的数据保存在gpu0上
    # 而VGGLoss在gpu3上，所以移动到gpu3来处理，之后再移动回去
    # KLDLoss和GANLoss并没有类似的操作，因为它们虽然是nn.Module
    # 但是本质上只是一个"计算"的封装，自身没有参数，
    # 也就是说kld_loss.cuda(2)之类的代码推测没什么用，还是在处理GPU0上的数据
    def forward(self, fake_imgs, real_imgs):
        if self.gpu:
            fake_imgs = fake_imgs.to(self.gpu)
            real_imgs = real_imgs.to(self.gpu)
        fake_vgg = self.vgg(fake_imgs)
        real_vgg = self.vgg(real_imgs)
        vgg_loss = 0
        for i in range(len(fake_vgg)):
            vgg_loss += self.weights[i] * self.criterion(fake_vgg[i], real_vgg[i].detach())
        return vgg_loss.cuda() if self.gpu else vgg_loss


# KL Divergence hinge_loss used in VAE with an image encoder
# VAE和KLD相关：https://toutiao.io/posts/387ohs/preview
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)


# GAN Feature Loss
# to help solve convergence and gradient vanish problems
class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred_fake, pred_real):
        feature_loss = 0
        # num_D是MultiScaleDiscriminator中的数量
        # 使用多个discriminator，进行corase-to-find的检测
        num_D = len(pred_fake)
        for i in range(num_D):
            for j in range(len(pred_fake[i]) - 1):  # for each layer output
                feature_loss += self.criterion(pred_fake[i][j], pred_real[i][j].detach())
        return feature_loss / num_D
