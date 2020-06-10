import numpy as np
from .train_util import get_float_tensor, get_device
from spade.dataloader.base_dataset import get_transform
from spade.dataloader.label_ids import all_labels
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def color2gray(colorlabel_numpy):
    rgb2gray_dict = {}
    for label in all_labels:
        [r, g, b] = label.color
        rgb2gray_dict[r * 256 * 256 + g * 256 + b] = label.id

    def convert_label(rgb):
        return rgb2gray_dict[rgb] if rgb in rgb2gray_dict else -1

    colorlabel_numpy = np.transpose(colorlabel_numpy, (2, 0, 1)).astype(int)
    rgblabel_numpy = colorlabel_numpy[0] * 256 * 256 + colorlabel_numpy[1] * 256 + colorlabel_numpy[2]
    vectorized_func = np.vectorize(convert_label)
    graylabel_numpy = vectorized_func(rgblabel_numpy).astype(np.uint8)
    return graylabel_numpy


def img2tensor(img_pil, opt):
    img_transform = get_transform(opt, False, normalize=True)
    img_tensor = img_transform(img_pil)
    return img_tensor


def colorlabel2tensor(colorlabel_pil, opt):
    global i
    colorlabel_numpy = np.asarray(colorlabel_pil)
    greylabel_numpy = color2gray(colorlabel_numpy)
    label_transform = get_transform(opt, False, normalize=False)
    label_tensor = label_transform(greylabel_numpy) * 255.0
    return label_tensor


def greylabel2tensor(greylabel_pil, opt):
    greylabel_numpy = np.asarray(greylabel_pil)
    label_transform = get_transform(opt, False, normalize=False)
    label_tensor = label_transform(greylabel_numpy) * 255.0
    return label_tensor


def centric_crop(image_pil, target_size):
    ow, oh = image_pil.size  # original width / height
    tw, th = target_size  # target width / height

    scale_ratio = max(tw / ow, th / oh)
    sw, sh = int(ow * scale_ratio), int(oh * scale_ratio)  # scaled width / height
    scaled_img = image_pil.resize((sw, sh), Image.BICUBIC)

    x = (sw - tw) / 2
    y = (sh - th) / 2
    croped_scaled_img = scaled_img.crop((x, y, x + tw, y + th))
    return croped_scaled_img


def load_img_tensor(img_pil, opt):
    device = get_device(opt)
    img_tensor = img2tensor(img_pil, opt)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor


def load_label_tensor(label_pil, opt):
    device = get_device(opt)
    if opt.use_greylabel:
        label_tensor = greylabel2tensor(label_pil, opt)
    else:
        label_tensor = colorlabel2tensor(label_pil, opt)
    label_tensor = label_tensor.long()
    label_tensor = label_tensor.unsqueeze(0).to(device)

    FloatTensor = get_float_tensor(opt)
    _, c, h, w = label_tensor.size()
    nc = opt.n_semantic
    input_label = FloatTensor(1, nc, h, w).zero_()
    label_tensor = input_label.scatter_(1, label_tensor, 1.0)
    return label_tensor
