import os
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


# 从数据集目录下读取文件，只读取训练/验证/测试集之一
# 返回二元元祖，分别为（内容图像路径的列表，标签图像路径的列表）
def get_data_paths(opt, sort=True, phase=None):
    root = opt.dataset_dir

    file_paths = load_files(root, only_img_file=True)
    img_paths = []
    label_paths = []
    for file_path in file_paths:
        if (phase is not None) and (phase not in file_path):
            continue
        if file_path.endswith('.png'):
            label_paths.append(file_path)
        elif file_path.endswith('.jpg'):
            img_paths.append(file_path)
    if sort:
        label_paths.sort()
        img_paths.sort()
    return label_paths, img_paths

