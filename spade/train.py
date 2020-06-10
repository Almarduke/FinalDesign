#%%

import time
import torch
import torch.nn as nn
from spade.dataloader import create_dataloader
from spade.options.train_options import TrainOptions
from spade.trainers.train_manager import TrainManager
from spade.util.visualizer import Visualizer
from spade.util.train_util import preprocess_train_data, get_device, get_gpu_ids, use_gpu
from spade.models.SpadeGAN import SpadeGAN
from spade.models.networks.loss import GANLoss, KLDLoss, VGGLoss, FeatureMatchingLoss


def main():
    opt = TrainOptions()
    dataloader = create_dataloader(opt)
    visualizer = Visualizer(opt)

    spade_gan = SpadeGAN(opt)
    gan_loss = GANLoss()
    kld_loss = KLDLoss()
    feature_loss = FeatureMatchingLoss()
    vgg_loss = VGGLoss()
    if use_gpu(opt):
        # https://www.zhihu.com/question/67726969/answer/389980788
        last_gpu = get_device(get_gpu_ids(opt)[-1])
        spade_gan = nn.DataParallel(spade_gan).cuda()
        gan_loss = gan_loss.cuda()
        kld_loss = kld_loss.cuda()
        feature_loss = feature_loss.cuda()
        vgg_loss = vgg_loss.move_to(last_gpu)

    trainer = TrainManager(opt, gan_loss, kld_loss, vgg_loss, feature_loss)
    optG, optD = trainer.create_optimizers(opt, spade_gan)

    for epoch in range(opt.current_epoch, opt.total_epochs):
        for batch_id, (label_imgs, real_imgs) in enumerate(dataloader):
            iter_start_time = time.time()
            seg_maps, real_imgs = preprocess_train_data(label_imgs, real_imgs, opt)

            # Generator优化一次
            optG.zero_grad()
            lossG, fake_imgs = trainer.get_lossG(seg_maps, real_imgs, spade_gan)
            lossG.backward()
            optG.step()

            # Discriminator优化一次
            optD.zero_grad()
            lossD = trainer.get_lossD(seg_maps, real_imgs, spade_gan)
            lossD.backward()
            optD.step()

            running_time = time.time() - iter_start_time
            visualizer.print_current_errors(epoch, batch_id, running_time, lossG, lossD)

            if batch_id % 100 == 0:
                visualizer.save_train_images(epoch, batch_id, label_imgs[0:2], real_imgs[0:2], fake_imgs[0:2])

        model = spade_gan.module if use_gpu(opt) else spade_gan
        model.save(epoch)
        trainer.update_learning_rate(epoch, optG=optG, optD=optD)

    print('Training was successfully finished.', flush=True)


if __name__ == '__main__':
    main()
