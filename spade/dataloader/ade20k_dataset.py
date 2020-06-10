from .base_dataset import BaseDataset
from .utils import get_data_paths


class ADE20KDataset(BaseDataset):
    # 数据集初始化时读取自己对应文件夹下的图片路径
    # 并检查路径是否正确（一张图像对应一张标签）
    def __init__(self, opt):
        super(ADE20KDataset, self).__init__(opt)
        label_paths, img_paths = get_data_paths(opt, sort=True, phase='train')
        self.pairing_check(opt, label_paths, img_paths)
        self.labels = label_paths
        self.imgs = img_paths
        self.dataset_size = len(self.labels)

    def __len__(self):
        return self.dataset_size
