# %%

import torch
import torch.nn as nn
from spade.options.test_options import TestOptions
from spade.util.visualizer import Visualizer
from spade.models.SpadeGAN import SpadeGAN
from spade.util.test_util import *
from spade.util.train_util import mkdir, tensor2img, get_device
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

opt = TestOptions()
spade_gan = SpadeGAN(opt)

if torch.cuda.is_available() > 0:
    # https://www.zhihu.com/question/67726969/answer/389980788
    spade_gan = nn.DataParallel(spade_gan).to(get_device(opt))


def spade_generate(label_image, style_image):
    label_tensor = load_label_tensor(label_image, opt)
    style_tensor = load_img_tensor(style_image, opt)
    fake_image = spade_gan(label_tensor, style_tensor).squeeze()
    fake_image = tensor2img(fake_image, opt)
    fake_image = Image.fromarray(fake_image).convert('RGB')
    return fake_image


def main():
    label_image = Image.open(opt.label_path)
    style_image = Image.open(opt.style_path)
    fake_image = spade_generate(label_image, style_image)
    fake_image.save(opt.result_path)
    print(f'Image saved at: {opt.result_path}', flush=True)


if __name__ == '__main__':
    main()

