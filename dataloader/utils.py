import os
from dataloader.ade20k_dataset import *
from torch.utils.data import DataLoader
from importlib import import_module


def is_img_file(file_name):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tiff', '.webp']
    return any(file_name.endswith(extension) for extension in IMG_EXTENSIONS)


# 递归地返回文件夹下所有文件的路径（包括子文件夹下的文件）
# 可以指定只返回图像文件
def load_files(root_dir, only_img_file=False):
    assert os.path.isdir(root_dir) or os.path.islink(root_dir), \
        f'"{root_dir}" is not a valid directory'
    file_paths = []
    for dir_path, dir_names, file_names in os.walk(root_dir):
        for file_name in file_names:
            if only_img_file and not is_img_file(file_name):
                continue
            filepath = os.path.join(dir_path, file_name)
            file_paths.append(filepath)
    return file_paths



