import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('citra.ui', self)
        self.Image = None
        self.actionLoad.clicked.connect(self.fungsi)
        self.pushButton.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStreching)
        self.actionnegative.triggered.connect(self.negative)
        self.actionOperasi_biner.triggered.connect(self.biner)
        self.actionHistogram_Grayscale.triggered.connect(self.histogramgrayscale)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogramClicked)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogramClicked)
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_derajat.triggered.connect(self.rotasi90derajat)
        self.action_90_derajat.triggered.connect(self.rotasimin90derajat)
        self.action45_derajat.triggered.connect(self.rotasi45derajat)
        self.action_45_derajat.triggered.connect(self.rotasimin45derajat)
        self.action180_derajat.triggered.connect(self.rotasi180derajat)
        self.action2x.triggered.connect(self.zoom2x)
        self.action3x.triggered.connect(self.zoom3x)
        self.action4x.triggered.connect(self.zoom4x)
        self.menuzoom_out_2.triggered.connect(self.zoomsatuperdua)
        self.action1_4.triggered.connect(self.zoomsatuperempat)
        self.action3_4.triggered.connect(self.zoomtigaperempat)
        self.action900x400.triggered.connect(self.dimensi900x400)
        self.actionCrop.triggered.connect(self.cropimage)
        self.actiontambah_dan_kurang.triggered.connect(self.aritmatika)
        self.actionkali_dan_bagi.triggered.connect(self.aritmatika2)
        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        self.actionOperasi_OR.triggered.connect(self.operasiOR)
        self.actionKonvolusi_A.triggered.connect(self.FilteringCliked)
        self.actionKonvolusi_B.triggered.connect(self.Filterring2)
        self.actionKernel_1_9.triggered.connect(self.Mean3x3)
        self.actionKernel_1_4.triggered.connect(self.Mean2x2)
        self.actionGaussian_Filter.triggered.connect(self.Gaussian)
        self.actionke_i.triggered.connect(self.Sharpening1)
        self.actionke_ii.triggered.connect(self.Sharpening2)
        self.actionke_iii.triggered.connect(self.Sharpening3)
        self.actionke_iv.triggered.connect(self.Sharpening4)
        self.actionke_v.triggered.connect(self.Sharpening5)
        self.actionke_vi.triggered.connect(self.Sharpening6)
        self.actionlaplace.triggered.connect(self.Laplace)
        self.actionMedian_Filter.triggered.connect(self.Median)
        self.actionMax_Filter.triggered.connect(self.Max)
        self.actionMin_Filter.triggered.connect(self.Min)
        # Tranformasi Fourier Diskrit
        self.actionDFT_Smoothing_Image.triggered.connect(self.SmothImage)
        self.actionDFT_Edge_Detection.triggered.connect(self.EdgeDetec)
        # Deteksi Tepi
        self.actionOperasi_Sobel.triggered.connect(self.Opsobel)
        self.actionOperasi_Prewitt.triggered.connect(self.Opprewitt)
        self.actionOperasi_Robert.triggered.connect(self.Oprobert)


    def displayImage(self, windows):
        qformat = QImage.Format_Indexed8
        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Image 1')
window.show()
sys.exit(app.exec_())