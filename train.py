#%%

import sys
import os
import time
import torch
import torch.nn as nn
import tqdm
from collections import OrderedDict
from dataloader import create_dataloader
from options.base_options import Options
from trainers.train_manager import TrainManager
from util.visualizer import Visualizer
from util.util import preprocess_train_data
from models.SpadeGAN import SpadeGAN
from models.networks.loss import GANLoss, KLDLoss, VGGLoss, GanFeatureLoss

#%%

opt = Options()

# os.environ['CUDA_VISIBLE_DEVICES'] = ",".join((str(gpu_id) for gpu_id in opt.gpu_ids))

# load the dataset
dataloader = create_dataloader(opt)

# create tool for counting iterations
# epoch_counter = EpochCounter(opt)

# create tool for visualization
visualizer = Visualizer(opt)

#%%

spade_gan = SpadeGAN(opt)
gan_loss = GANLoss()
kld_loss = KLDLoss()
vgg_loss = VGGLoss(gpu_id=3)
gan_feature_loss = GanFeatureLoss()

if torch.cuda.is_available() > 0:
    # https://www.zhihu.com/question/67726969/answer/389980788
    spade_gan = nn.DataParallel(spade_gan).cuda()
    gan_loss = gan_loss.cuda()
    kld_loss = kld_loss.cuda()
    vgg_loss = vgg_loss.cuda(3)
    gan_feature_loss = gan_feature_loss.cuda()


# create trainer for our model
trainer = TrainManager(opt, gan_loss, kld_loss, vgg_loss, gan_feature_loss)
optG, optD = trainer.create_optimizers(opt, spade_gan)

#%%

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

        if batch_id % 200 == 0:
            visualizer.save_images(epoch, batch_id, label_imgs, real_imgs, fake_imgs)

    spade_gan.module.save(epoch)
    trainer.update_learning_rate(epoch, optG=optG, optD=optD)

print('Training was successfully finished.', flush=True)

#%%
