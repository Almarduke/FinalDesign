import numpy as np
import json
from label_ids import all_labels


def convert_coco2custom(coco_segmap):
    coco2custom_dict = {}
    for label in all_labels:
        for coco_id in label.coco_ids:
            coco2custom_dict[coco_id] = label.id

    def convert_label(coco_label_id):
        try:
            return coco2custom_dict[coco_label_id]
        except KeyError:
            return 0  # zero means "don't care"

    vectorized_func = np.vectorize(convert_label)
    return vectorized_func(coco_segmap)
