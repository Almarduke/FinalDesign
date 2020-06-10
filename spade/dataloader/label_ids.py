import numpy as np


class Unknown:
    id = 0
    coco_ids = []
    color = [0, 0, 0]


class Sky:
    id = 1
    coco_ids = [156]
    color = [216, 240, 240]


class Cloud:
    id = 2
    coco_ids = [105]
    color = [252, 252, 252]


class Water:
    id = 3
    coco_ids = [147, 177]
    color = [0, 191, 255]


class Ocean:
    id = 4
    coco_ids = [154]
    color = [0, 0, 205]


class Tree:
    id = 5
    coco_ids = [93, 128, 168]
    color = [69, 139, 0]


class Bush:
    id = 6
    coco_ids = [96]
    color = [102, 205, 0]


class Grass:
    id = 7
    coco_ids = [123]
    color = [154, 205, 50]


class Snow:
    id = 8
    coco_ids = [158]
    color = [230, 230, 250]


class Mountain:
    id = 9
    coco_ids = [126, 134]
    color = [139, 90, 43]


class Rock:
    id = 10
    coco_ids = [149, 161]
    color = [190, 190, 190]


class Earth:
    id = 11
    coco_ids = [110, 135]
    color = [139, 115, 85]


class Sand:
    id = 12
    coco_ids = [153]
    color = [255, 231, 186]


all_labels = sorted([
    Unknown,
    Sky, Cloud, Water, Ocean,
    Tree, Bush, Grass, Snow,
    Mountain, Rock, Earth, Sand
], key=lambda x: x.id)

natural_colormap = np.asarray([label.color for label in all_labels])

