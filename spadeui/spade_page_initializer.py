from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import *
from PyQt5 import uic
import copy
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def init_actions(spade_page):
    spade_page.ui.withdrawAction.triggered.connect(spade_page.withdraw_painting)
    spade_page.ui.clearAction.triggered.connect(spade_page.reset_label_canvas)
    spade_page.ui.fillAction.triggered.connect(spade_page.set_fill_mode)
    spade_page.ui.openAction.triggered.connect(spade_page.open_label)
    spade_page.ui.saveAction.triggered.connect(spade_page.save_file)
    spade_page.ui.generateButton.pressed.connect(spade_page.generate_image)


def init_label_canvas(spade_page):
    canvas = spade_page.ui.labelCanvas
    canvas.mousePressEvent = spade_page.brush_press_event
    canvas.mouseMoveEvent = spade_page.brush_move_event
    canvas.mouseReleaseEvent = spade_page.brush_release_event
    spade_page.reset_label_canvas()


def init_brushzise_selector(spade_page):
    brushsize_icon = QLabel()
    brushsize_icon.setFixedSize(18, 18)
    icon = QPixmap(os.path.join('./spadeui/resource/toolbuttons', 'brushsize.png')).scaled(16, 16)
    brushsize_icon.setPixmap(icon)

    brushsize_selector = QSlider()
    brushsize_selector.setFixedSize(120, 32)
    brushsize_selector.setRange(2, 48)
    brushsize_selector.setValue(spade_page.config.brush_size)
    brushsize_selector.setOrientation(Qt.Horizontal)
    brushsize_selector.valueChanged.connect(lambda s: spade_page.config.set_brush_size(s))

    spade_page.ui.brushSizeToolbar.addWidget(brushsize_icon)
    spade_page.ui.brushSizeToolbar.addWidget(brushsize_selector)


def init_mode_toolbar(spade_page):
    pass
    # generate_mode = QRadioButton()
    # generate_mode.setText('语义生成')
    # generate_mode.setChecked(True)
    # generate_mode.pressed.connect(lambda: spade_page.config.set_mode(NGMode.label_generate))
    # transfer_mode = QRadioButton()
    # transfer_mode.setText('风格迁移')
    # transfer_mode.pressed.connect(lambda: spade_page.config.set_mode(NGMode.style_transfer))
    # spade_page.ui.modeToolbar.addWidget(generate_mode)
    # spade_page.ui.modeToolbar.addWidget(transfer_mode)


def init_save_toolbar(spade_page):
    save_config = QCheckBox()
    save_config.setText('同时保存语义标签')
    save_config.pressed.connect(spade_page.config.set_save_config)
    spade_page.ui.saveToolbar.addWidget(save_config)


def init_brush_buttons(spade_page):
    def button_method(x):
        return lambda: spade_page.select_brush_button(x)

    for brush_button, _, brush_id in spade_page.brush_buttons:
        brush_button.pressed.connect(button_method(brush_id))
    spade_page.select_brush_button(spade_page.config.brush_id)


def init_style_buttons(spade_page):
    set_style_layout(spade_page)
    for button_id, img_url in enumerate(spade_page.config.style_image_urls):
        insert_style_button(spade_page, img_url, button_id)
    spade_page.select_style_button(spade_page.config.style_id)
    spade_page.add_style_button.pressed.connect(spade_page.add_style_image)
    spade_page.scrollLayout.addWidget(spade_page.add_style_button)


def set_style_layout(spade_page):
    scrollContent = spade_page.ui.styleContainer
    scrollLayout = QHBoxLayout(scrollContent)
    scrollLayout.setContentsMargins(0, 10, 0, 10)
    scrollLayout.setSpacing(36)
    scrollContent.setLayout(scrollLayout)
    spade_page.scrollLayout = scrollLayout


def insert_style_button(spade_page, img_url, button_id):
    style_button = get_style_button(img_url)
    style_button.pressed.connect(lambda: spade_page.select_style_button(button_id))
    spade_page.scrollLayout.addWidget(style_button)
    spade_page.style_buttons.append(style_button)

    image = Image.open(img_url).convert("RGB")
    image = image.resize((512, 512), Image.ANTIALIAS)
    spade_page.style_images.append(image)


def get_style_button(img_url, size=(92, 92)):
    style_button = QPushButton()
    style_button.setFixedSize(100, 100)
    style_button.setIcon(QIcon(QPixmap(img_url)))
    style_button.setIconSize(QSize(*size))
    return style_button


def init_generate_canvas(self):
    canvas = self.ui.generateCanvas
    canvas.setPixmap(QPixmap(512, 512))
    p = QPainter(canvas.pixmap())
    gray = QColor('#EEEEEE')
    p.setPen(QPen(gray, 2))
    p.setBrush(gray)
    p.drawRect(0, 0, 512, 512)
    canvas.update()