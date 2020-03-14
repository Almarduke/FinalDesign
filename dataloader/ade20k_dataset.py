"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from PIL import Image
from .base_dataset import BaseDataset
from .utils import load_files
from options import ADE20K_DATASET_OPTION

class ADE20KDataset(BaseDataset):
    option = ADE20K_DATASET_OPTION

    def __init__(self, opt):
        super(ADE20KDataset, self).__init__()
        label_paths, img_paths = self.get_paths(opt, sort=True)
        if opt.pairing_check:
            for img_path, label_path in zip(label_paths, img_paths):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                label_name = os.path.splitext(os.path.basename(label_path))[0]

                assert img_name == label_name,\
                    "The label-image pair seems to be wrong since file names are different."

        self.imgs = img_paths
        self.labels = label_paths
        self.dataset_size = len(self.labels)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)

        return parser

    def get_paths(self, opt, sort=True):
        root = opt.dataroot
        phase = opt.phase

        file_paths = load_files(root, only_img_file=True)
        img_paths = []
        label_paths = []
        for file_path in file_paths:
            if phase not in img_paths:
                continue
            if file_path.endswith('.jpg'):
                img_paths.append(file_path)
            elif file_path.endswith('.png'):
                label_paths.append(file_path)
        if sort:
            img_paths.sort()
            label_paths.sort()
        return img_paths, label_paths

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc

    def __len__(self):
        return self.dataset_size

