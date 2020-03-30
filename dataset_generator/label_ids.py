class Unknown:
    def __init__(self):
        self.new_id = 0
        self.ade_ids = []
        self.color = [255, 255, 255]


class Sky:
    def __init__(self):
        self.new_id = 1
        self.ade_ids = [3]
        self.color = [240, 255, 255]


class Water:
    def __init__(self):
        self.new_id = 2
        self.ade_ids = [22, 61, 169]
        self.color = [0, 191, 255]


class Sea:
    def __init__(self):
        self.new_id = 3
        self.ade_ids = [27]
        self.color = [0, 0, 205]


class Waterfall:
    def __init__(self):
        self.new_id = 4
        self.ade_ids = [114]
        self.color = [230, 230, 250]


class Tree:
    def __init__(self):
        self.new_id = 5
        self.ade_ids = [5]
        self.color = [102, 205, 0]


class Grass:
    def __init__(self):
        self.new_id = 6
        self.ade_ids = [10, 18]
        self.color = [154, 205, 50]


class Mountain:
    def __init__(self):
        self.new_id = 7
        self.ade_ids = [17, 69]
        self.color = [139, 90, 43]


class Rock:
    def __init__(self):
        self.new_id = 8
        self.ade_ids = [35]
        self.color = [190, 190, 190]


class Earth:
    def __init__(self):
        self.new_id = 9
        self.ade_ids = [14]
        self.color = [139, 115, 85]


class Sand:
    def __init__(self):
        self.new_id = 10
        self.ade_ids = [47]
        self.color = [255, 231, 186]


all_labels = [Unknown(), Sky(), Water(), Sea(), Waterfall(), Tree(), Grass(), Mountain(), Rock(), Earth(), Sand()]
label_colormap = [label.color for label in all_labels]
