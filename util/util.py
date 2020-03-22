"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import importlib
import torch
import numpy as np
from PIL import Image
import os
import argparse
import dill as pickle


def tensor2img(image_tensor, opt, imtype=np.uint8, normalize=True):
    img_np = image_tensor.detach().cpu().float().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # 某个nparray的形状为[n, c, w, h] => [c, w, h]，转成[w, h, c]
    img_np = ((img_np * opt.img_var + opt.img_mean) if normalize else img_np) * 255.0
    img_np = np.clip(img_np, 0, 255)
    return img_np.astype(imtype)


# Converts grey image into a colorful label map
def tensor2label(label_tensor, opt, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float().numpy()
    label_numpy = colorize(label_tensor, opt.n_label)
    label_numpy = np.transpose(label_numpy, (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(img_numpy, img_dir, img_name):
    img_dir = mkdir(img_dir)
    img_path = os.path.join(img_dir, img_name)
    img_pil = Image.fromarray(img_numpy)
    img_pil.save(img_path)


def save_network(net, label, epoch, opt):
    filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = mkdir(os.path.join(opt.checkpoints_dir, opt.dataset))
    save_path = os.path.join(save_dir, filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.dataset)
    save_path = os.path.join(save_dir, save_filename)
    if os.path.exists(save_path):
        weights = torch.load(save_path)
        net.load_state_dict(weights)
    return net


# 将无符号整数（掩膜的编号）转换为二进制形式
# 编号最多为0-255
def uint8_to_binary(n, count=8):
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def label_colormap(n_label):
    color_map = {}
    for label_id in range(1, n_label + 1):
        reversed_binary_id = uint8_to_binary(label_id)[::-1]
        r = int(reversed_binary_id[2:] + '00', 2)
        g = int(reversed_binary_id[1:7] + '00', 2)
        b = int(reversed_binary_id[:6] + '00', 2)
        color_map[label_id] = [r, g, b]
    return color_map


# 为不同id的标签添加不同颜色的mask
# 如果有150个标签，那么id=1~150
# 无法识别的对应id=0，颜色选择白色
def colorize(gray_image, n_label):
    cmap = label_colormap(n_label)
    _, w, h = gray_image.shape
    color_image = torch.ByteTensor(3, w, h).fill_(255)
    for label_id in cmap.keys():
        mask = (label_id == gray_image[0])
        color_image[0][mask] = cmap[label_id][0]
        color_image[1][mask] = cmap[label_id][1]
        color_image[2][mask] = cmap[label_id][2]
    return color_image.numpy()


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


# 处理XXXDataset.__getitem__的输出结果
# 把label图像变成one-hot的tensor，并且把tensor移到GPU
# 返回(语义图tensor，图像tensor)
def preprocess_train_data(label_tensor, img_tensor, opt):
    assert label_tensor.size()[1] == 1 and img_tensor.size()[1] == 3, \
        '请检查tensor的通道数，label_tensor应该为1，img_tensor应该为3'

    # move to GPU and change data types
    label_tensor = label_tensor.long()
    if use_gpu(opt):
        label_tensor = label_tensor.cuda()
        img_tensor = img_tensor.cuda()

    # one-hot label map
    # https://pytorch.org/docs/stable/tensors.html  # torch.Tensor.scatter_
    # scatter_(dim, index, src) → Tensor
    # self[i][index[i][j][k][L]][k][L] = src[i][j][k][L]  # if dim == 1
    # 在某个维度上进行scatter，就是在该维度上进行one-hot
    # 在dim=1上进行one-hot，对于label_map做one-hot，结果存放在了input_label
    FloatTensor = get_float_tensor(opt)
    bs, _, h, w = label_tensor.size()
    nc = opt.n_semantic
    input_label = FloatTensor(bs, nc, h, w).zero_()
    seg_maps = input_label.scatter_(1, label_tensor, 1.0)
    seg_maps.requires_grad_(False)
    return seg_maps, img_tensor


def get_float_tensor(opt):
    FloatTensor = torch.cuda.FloatTensor if use_gpu(opt) else torch.FloatTensor
    return FloatTensor


def use_gpu(opt):
    return len(opt.gpu_ids) > 0
