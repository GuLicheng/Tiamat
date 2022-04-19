import os
from PIL import Image
import numpy as np
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
import cv2 as cv
from PyQt6 import QtCore, QtGui, QtWidgets
import sys
sys.path.append(r"D:\code\Tiamat")
from lv_py.utils.logger import Logger

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(280, 460, 201, 23))
        self.pushButton.setObjectName("pushButton")

        self.recordButton = QtWidgets.QPushButton(self.centralwidget)
        self.recordButton.setGeometry(QtCore.QRect(0, 0, 201, 23))
        self.recordButton.setObjectName("record")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(210, 90, 300, 300))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "通过数组显示图片"))
        self.recordButton.setText(_translate("MainWindow", "Save"))
        self.label.setText(_translate("MainWindow", "TextLabel"))


class My_UI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle('窗口标题')
        self.mask = np.load("refined_attention_masks.npy", allow_pickle=True).item()
        self.filenames = list(self.mask.keys())
        self.idx = 0

        self.saliency = r"D:\experiment\saliency\saliency_mask"
        self.semantic = r"D:\dataset\VOCdevkit\VOC2012\SegmentationClass"
        self.image = r"D:\dataset\VOCdevkit\VOC2012\JPEGImages"

        self.logger = Logger("./")
        


    def pushbuttonthing(self):

        # attention mask

        name = self.filenames[self.idx]
        img_src = self.mask[name] # [6, H, W]
        masks = [mask * 255 for mask in img_src]
        masks = np.hstack(masks)
        img_src = cv.cvtColor(masks, cv.COLOR_GRAY2BGR)

        # image, gt, saliency from feature

        image = cv.imread(f"{os.path.join(self.image, name)}.jpg")
        semantic = np.array(Image.open(f"{os.path.join(self.semantic, name)}.png"))
        semantic[semantic != 0] = 255
        semantic = cv.cvtColor(semantic, cv.COLOR_GRAY2RGB)
        saliency = np.array(Image.open(f"{os.path.join(self.saliency, name)}.png"))
        saliency = cv.cvtColor(saliency, cv.COLOR_GRAY2RGB)

        img2 = np.hstack([image, semantic, saliency])
        cv.imshow(f"{name}", img_src)
        cv.imshow("", img2)

        cv.waitKey(0)
        self.idx += 1

    def record(self):

        self.logger.info(f"{self.filenames[self.idx - 1]}")

    def run(self):
        self.pushButton.clicked.connect(self.pushbuttonthing)
        self.recordButton.clicked.connect(self.record)

    


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 显示窗口
    win = My_UI()
    win.show()
    win.run()
    sys.exit(app.exec())
