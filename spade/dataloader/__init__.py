import importlib
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset


def find_dataset_class(dataset_name):
    dataset_name = dataset_name.split('_')[0]
    dataset_filename = "spade.dataloader." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'

    dataset = None
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls
    if dataset is None:
        raise ValueError('数据集名称错误，找不到对应的类：数据集名称应该为"ade20k"或者"natural"')

    return dataset


def create_dataloader(opt):
    dataset_class = find_dataset_class(opt.dataset)
    dataset = dataset_class(opt)
    print(f"dataset '{opt.dataset}' was created", flush=True)
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=opt.is_train
    )
