import numpy as np
import json
from label_ids import all_labels


def get_label_convert_dict():
    convert_dict = {}
    for label in all_labels:
        for ade_id in label.ade_ids:
            convert_dict[ade_id] = label.new_id
    return convert_dict


label_convert_dict = get_label_convert_dict()

print(label_convert_dict)


def convert_segmap(ade_segmap):
    def convert_label(ade_label_id):
        try:
            return label_convert_dict[ade_label_id]
        except KeyError:
            return 0  # zero means "don't care"

    vectorized_func = np.vectorize(convert_label)
    return vectorized_func(ade_segmap)
