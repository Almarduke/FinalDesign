from PyQt5.QtGui import QColor
from spade.dataloader.label_ids import *


class NGMode:
    Paint = 'Paint'
    Fill = 'Fill'


class NGConfig:
    @staticmethod
    def brush_color_of_id(brush_id):
        def hex_color(color_channels):
            r, g, b = color_channels
            hex_bits = '0123456789ABCDEF'
            return '#' + ''.join([hex_bits[c // 16] + hex_bits[c % 16] for c in [r, g, b]])

        brush_colors = natural_colormap
        return QColor(hex_color(brush_colors[brush_id]))

    def __init__(self):
        self.image_size = (512, 512)  # 语义分割图、风格图像、生成图像的大小
        self.brush_size = 24  # 绘画时笔刷的粗细
        self.brush_id = 1  # 笔刷的种类（12种）
        self.style_id = 0  # 风格图像的id
        self.mode = NGMode.Paint  # 绘制还是填充
        self.save_label = False  # 是否同时保存标签图
        self.brush_pos = None  # 笔刷的位置
        self.style_image_urls = [f'spadeui/resource/style-image/style{i}.png' for i in range(7)]  # 风格图像的路径
        self.add_style_button_url = f'spadeui/resource/style-image/add.png'  # add按钮的图像路径

    @property
    def brush_color(self):
        return NGConfig.brush_color_of_id(self.brush_id)

    def set_brush_size(self, brush_size):
        self.brush_size = brush_size

    def set_mode(self, mode):
        self.mode = mode

    def set_brush_id(self, brush_id):
        self.brush_id = brush_id

    def set_style_id(self, style_id):
        self.style_id = style_id

    def set_save_config(self):
        self.save_label = not self.save_label
