from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QBuffer, QPoint
from PyQt5.QtGui import *
from PyQt5 import uic
import io

from spade.dataloader.label_ids import *
from spade.util.test_util import *
from spade.test import spade_generate
from spadeui.spade_page_utils import is_label_image
from spadeui.ngconfig import NGConfig, NGMode
import spadeui.spade_page_initializer as initializer

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SpadePage:
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit

        self.config = NGConfig()
        self.ui = uic.loadUi('spadeui/spade.ui')
        self.ui.setWindowFlags(
            Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint |
            Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        self.brush_buttons = [
            (self.ui.skyIcon, self.ui.skyBrush, Sky.id), (self.ui.cloudIcon, self.ui.cloudBrush, Cloud.id),
            (self.ui.waterIcon, self.ui.waterBrush, Water.id), (self.ui.oceanIcon, self.ui.oceanBrush, Ocean.id),
            (self.ui.treeIcon, self.ui.treeBrush, Tree.id), (self.ui.bushIcon, self.ui.bushBrush, Bush.id),
            (self.ui.grassIcon, self.ui.grassBrush, Grass.id), (self.ui.snowIcon, self.ui.snowBrush, Snow.id),
            (self.ui.mountainIcon, self.ui.mountainBrush, Mountain.id), (self.ui.rockIcon, self.ui.rockBrush, Rock.id),
            (self.ui.earthIcon, self.ui.earthBrush, Earth.id), (self.ui.sandIcon, self.ui.sandBrush, Sand.id)
        ]
        self.style_buttons = []
        self.style_images = []
        self.pix_backup = []
        self.add_style_button = initializer.get_style_button(self.config.add_style_button_url, size=(60, 60))
        self.init_spade()

    ################ 初始化界面上各个组建的方法 ################

    def init_spade(self):
        initializer.init_actions(self)
        initializer.init_label_canvas(self)
        initializer.init_generate_canvas(self)
        initializer.init_brush_buttons(self)
        initializer.init_brushzise_selector(self)
        initializer.init_mode_toolbar(self)
        initializer.init_save_toolbar(self)
        initializer.init_style_buttons(self)

    ################ 处理和笔刷相关的事件 ################

    def brush_press_event(self, e):
        self.pix_backup.append(self.ui.labelCanvas.pixmap().copy())
        if self.config.mode == NGMode.Paint:
            self.config.brush_pos = e.pos()
            self.draw_segmap(e)
        elif self.config.mode == NGMode.Fill:
            self.fill_segmap(e)

    def brush_move_event(self, e):
        if self.config.brush_pos:
            self.draw_segmap(e)

    def brush_release_event(self, e):
        self.config.brush_pos = None

    def draw_segmap(self, e):
        p = QPainter(self.ui.labelCanvas.pixmap())
        p.setPen(QPen(self.config.brush_color, self.config.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if e.pos() == self.config.brush_pos:
            p.drawPoint(e.pos())
        else:
            p.drawLine(self.config.brush_pos, e.pos())
        self.config.brush_pos = e.pos()
        self.ui.labelCanvas.update()

    def fill_segmap(self, e):
        canvas = self.ui.labelCanvas
        pixmap = canvas.pixmap()
        image = pixmap.toImage()
        w, h = image.width(), image.height()
        x, y = e.x(), e.y()

        to_fill = set()
        to_check = {(x, y)}
        fill_color = QColor(image.pixel(x, y))

        while to_check:
            ox, oy = to_check.pop()
            to_fill.add((ox, oy))
            for move_x, move_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = ox + move_x, oy + move_y
                if 0 <= nx < w and 0 <= ny < h \
                        and (nx, ny) not in to_fill \
                        and QColor(image.pixel(nx, ny)) == fill_color:
                    to_check.add((nx, ny))

        p = QPainter(pixmap)
        p.setPen(QPen(self.config.brush_color))
        p.drawPoints(*[QPoint(*pos) for pos in to_fill])
        canvas.update()

    def select_brush_button(self, selected_id):
        self.config.brush_id = selected_id
        for _, brush_widget, brush_id in self.brush_buttons:
            opacity = 1 if brush_id == selected_id else 0.2
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(opacity)
            brush_widget.setGraphicsEffect(effect)

    ################ 处理和底部风格图像相关的方法 ################

    def select_style_button(self, selected_id):
        self.config.style_id = selected_id
        for button_id, style_button in enumerate(self.style_buttons):
            if button_id == selected_id:
                border_style = 'border: 4px solid #FF4500;border-radius: 4px;'
            else:
                border_style = 'border: 4px solid #CCCCCC;border-radius: 4px;'
            style_button.setStyleSheet(border_style)

    def add_style_image(self):
        path, _ = QFileDialog.getOpenFileName(self.ui.labelCanvas, "上传风格图像", "",
                                              "PNG image files (*.png);;JPEG image files (*jpg)")
        if path:
            button_id = len(self.style_buttons)
            self.scrollLayout.removeWidget(self.add_style_button)
            initializer.insert_style_button(self, path, button_id)
            self.scrollLayout.addWidget(self.add_style_button)
            self.select_style_button(self.config.style_id)

    ################ 顶部工具栏的各个按钮的方法 ################

    def withdraw_painting(self):
        if self.pix_backup:
            last_pix = self.pix_backup.pop()
            self.ui.labelCanvas.setPixmap(last_pix)
            self.ui.labelCanvas.update()

    def reset_label_canvas(self):
        self.pix_backup.clear()

        labelcanvas = self.ui.labelCanvas
        labelcanvas.setPixmap(QPixmap(512, 512))
        p = QPainter(labelcanvas.pixmap())
        p.setPen(QPen(QColor(NGConfig.brush_color_of_id(Sky.id)), 2))
        p.setBrush(QColor(NGConfig.brush_color_of_id(Sky.id)))
        p.drawRect(0, 0, 512, 288)
        p.setPen(QPen(QColor(NGConfig.brush_color_of_id(Ocean.id)), 2))
        p.setBrush(QColor(NGConfig.brush_color_of_id(Ocean.id)))
        p.drawRect(0, 288, 512, 512)
        labelcanvas.update()

        generate_canvas = self.ui.generateCanvas
        generate_canvas.setPixmap(QPixmap(512, 512))
        p = QPainter(generate_canvas.pixmap())
        p.setPen(QPen(QColor('#E8E8E8'), 2))
        p.setBrush(QColor('#E8E8E8'))
        p.drawRect(0, 0, 512, 512)
        generate_canvas.update()

    def set_fill_mode(self):
        if self.ui.fillAction.isChecked():
            self.config.mode = NGMode.Fill
        else:
            self.config.mode = NGMode.Paint

    def open_label(self):
        path, _ = QFileDialog.getOpenFileName(self.ui.labelCanvas, "打开语义分割图", "",
                                              "PNG image files (*.png);;JPEG image files (*jpg)")
        while path:
            colorlabel = Image.open(path)
            colorlabel = centric_crop(colorlabel, (512, 512))
            colorlabel = colorlabel.convert("RGB")
            if is_label_image(colorlabel):
                self.draw_image_to_canvas(colorlabel, self.ui.labelCanvas)
                break
            else:
                reply = QMessageBox.critical(
                    self.ui, '无法打开文件', '不是合法的语义分割图',
                    QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok
                )
                if reply == QMessageBox.Cancel:
                    break
                else:
                    path, _ = QFileDialog.getOpenFileName(
                        self.ui.labelCanvas, "上传语义分割图", "",
                        "PNG image files (*.png);;JPEG image files (*jpg)"
                    )

    def save_file(self):
        images = [('生成图像', self.ui.generateCanvas)]
        if self.config.save_label:
            images.append(('语义分割图', self.ui.labelCanvas))

        saved_images = []
        for name, canvas in images:
            path, _ = QFileDialog.getSaveFileName(canvas, f'保存{name}', name, "PNG Image file (*.png)")
            if path:
                pixmap = canvas.pixmap()
                pixmap.save(path, 'PNG')
                saved_images.append(name)

        if len(saved_images) > 0:
            saved_images = '和'.join(saved_images)
            QMessageBox.about(self.ui, '保存成功', f'{saved_images}保存成功')

    #################### 生成图像的方法 #####################

    def generate_image(self):
        label_canvas = self.ui.labelCanvas
        qimage = label_canvas.pixmap().toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        qimage.save(buffer, "PNG")
        colorlabel = Image.open(io.BytesIO(buffer.data()))
        colorlabel = colorlabel.convert('RGB')

        style_image = self.style_images[self.config.style_id]

        generated_image = spade_generate(colorlabel, style_image)
        generate_canvas = self.ui.generateCanvas
        self.draw_image_to_canvas(generated_image, generate_canvas)

    def draw_image_to_canvas(self, image, canvas):
        image_bytes = image.tobytes("raw", "RGB")
        qimage = QImage(image_bytes, 512, 512, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        canvas.setPixmap(pixmap)
        canvas.update()
