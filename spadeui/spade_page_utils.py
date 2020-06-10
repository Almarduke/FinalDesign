import numpy as np
from spade.util.test_util import color2gray


def is_label_image(colorlabel):
    colorlabel_numpy = np.asarray(colorlabel)
    graylabel_numpy = color2gray(colorlabel_numpy)
    return 0 not in np.unique(graylabel_numpy)
