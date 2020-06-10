from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from spadeui.spade_page import SpadePage

if __name__ == '__main__':
    app = QApplication([])
    spadepage = SpadePage()
    spadepage.ui.show()
    app.exec_()
