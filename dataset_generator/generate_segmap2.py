import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import time

import tensorflow as tf
from os import listdir
from os.path import isfile, join


##########################################

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
label_dir = '/chenxiao/FinalDesign/dataset/Nature/labels/'
img_dir = '/chenxiao/FinalDesign/dataset/Nature/images/'
img_files = sorted([f for f in listdir(img_dir) if isfile(join(img_dir, f))])[:5000]


##########################################

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 512
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


##########################################

model_dir = '/chenxiao/FinalDesign/dataset/Nature'
_TARBALL_NAME = 'deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'

download_path = os.path.join(model_dir, _TARBALL_NAME)
MODEL = DeepLabModel(download_path)
print('model loaded successfully!')


##########################################

for index, img_file in enumerate(img_files):
    iter_start_time = time.time()
    original_im = Image.open(f'{img_dir}/{img_file}')
    resized_im, seg_map = MODEL.run(original_im)
    seg_image = seg_map.astype(np.uint8)
    image_pil = Image.fromarray(seg_image)
    image_pil = image_pil.resize(original_im.size, Image.ANTIALIAS)
    image_pil.save(f'{label_dir}/{img_file}')
    running_time = time.time() - iter_start_time
    print(f'{index}: {running_time} sec')