import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from skimage.metrics import structural_similarity
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtGui import *


class ShowImage(QtWidgets.QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('citra.ui', self)
        self.Image = None
        self.actionLoad.triggered.connect(self.openClicked)
        self.actionSave.triggered.connect(self.save)
        self.pushButton_3.clicked.connect(self.grayscale)
        self.pushButton.clicked.connect(self.ssim100)
        self.pushButton_2.clicked.connect(self.ssim50)
        self.verticalSlider.valueChanged.connect(self.Brightness)
        self.actionopsobel.triggered.connect(self.Opsobel)
        self.detect100baru.clicked.connect(self.ssim100baru)
        self.detect50baru.clicked.connect(self.ssim50baru)



    def openClicked(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg)")#membuka file gambar
        if filename:#jika file gambar ada
            self.Image = cv2.imread(filename)#membaca file gambar
            self.displayImage(1)#menampilkan gambar

    def save(self):#fungsi menyimpan gambar
        if self.Image is not None:#jika gambar ada
            filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.xpm *.jpg)")#menyimpan gambar
            if filename:#jika gambar tersimpan
                cv2.imwrite(filename,self.Image)#menyimpan gambar

    def grayscale(self):#fungsi grayscale
        H, W = self.Image.shape[:2]#membaca ukuran tinggi dan lebar citra
        gray = np.zeros((H, W), np.uint8)#membuat matriks kosong dengan ukuran tinggi dan lebar citra
        for i in range(H):#mengatur pergerakan citra
            for j in range(W):#mengatur pergerakan citra
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] + 0.587
                                     * self.Image[i, j, 1] + 0.114 * self.Image[i, j, 2], 0, 255)#menghitung nilai grayscale
        self.Image = gray#menyimpan nilai grayscale
        self.displayImage(2)#menampilkan gambar

    def Konvolusi(self, X,F): #Fungsi konvolusi 2D
        X_height = X.shape[0] #membaca ukuran tinggi dan lebar citra
        X_width = X.shape[1]

        F_height = F.shape[0] #membaca ukuran tinggi dan lebar kernel
        F_width = F.shape[1]#membaca ukuran tinggi dan lebar kernel

        H = (F_height) // 2#mengatur pergerakan karnel
        W = (F_width) // 2#mengatur pergerakan karnel

        out = np.zeros((X_height, X_width))#membuat matriks kosong dengan ukuran tinggi dan lebar citra

        for i in np.arange(H + 1, X_height - H): #mengatur pergerakan karnel
            for j in np.arange(W + 1, X_width - W):#mengatur pergerakan karnel
                sum = 0#membuat variabel sum
                for k in np.arange(-H, H + 1):#mengatur pergerakan karnel
                    for l in np.arange(-W, W + 1):#mengatur pergerakan karnel
                        a = X[i + k, j + l] #menampung nilai pixel
                        w = F[H + k, W + l]#menampung nilai kernel
                        sum += (w * a) #menampung nilai total perkalian w kali a
                out[i,j] = sum  #menampung hasil
        return out#mengembalikan nilai out

    def Opsobel(self):#fungsi operasi sobel
        # mendefinisikan variabel img yang diinisialisasi dengan hasil konversi gambar dari warna ke grayscale.
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)#mengubah citra ke grayscale
        # X dan Y adalah matriks filter Sobel pada sumbu X dan Y.
        X = np.array([[-1, 0, 1],#membuat matriks filter sobel
                      [-2, 0, 2],
                      [-1, 0, 1]])
        Y = np.array([[-1, -2, -1],#membuat matriks filter sobel
                      [0, 0, 0],
                      [1, 2, 1]])
        # img_Gx dan img_Gy adalah citra hasil konvolusi dari matriks filter X dan Y terhadap citra grayscale img.
        img_Gx = self.Konvolusi(img, X)
        img_Gy = self.Konvolusi(img, Y)
        # img_out adalah citra hasil kombinasi citra edge detection pada sumbu X dan Y.
        img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
        # setelah mendapatkan citra img_out, citra tersebut di-normalisasi dengan membagi dengan nilai maksimal dan dikalikan dengan 255.
        img_out = (img_out / np.max(img_out)) * 255
        print('---Nilai Pixel Operasi Sobel--- \n', img_out)#menampilkan nilai pixel
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')#menampilkan gambar
        plt.show()#menampilkan gambar

    def Brightness(self, value):  # fungsi untuk mengatur kecerahan
        try: #mengubah gambar menjadi grayscale
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except: #exception jika gambar sudah grayscale maka tidak perlu diubah menjadi grayscale
            pass  #pass adalah perintah untuk melanjutkan ke perintah selanjutnya

        H, W = self.Image.shape[:2] #mengambil nilai tinggi dan lebar gambar
        cahaya = value + 30 #mengambil nilai kecerahan
        for i in range(H): #perulangan untuk mengatur kecerahan
            for j in range(W): #perulangan untuk mengatur kecerahan
                a = self.Image.item(i, j) #mengambil nilai pixel
                b = np.clip(a + cahaya, 0, 255) #rumus kecerahan
                if b > 255: #jika nilai pixel lebih dari 255 maka nilai pixel adalah 255
                    b = 255 #jika b sama dengan 255 maka nilai pixel adalah 255
                elif b < 0: #jika nilai pixel kurang dari 0 maka nilai pixel adalah 0
                    b = 0 #jika b sama dengan 0 maka nilai pixel adalah 0
                else: #jika nilai pixel berada diantara 0 dan 255 maka nilai pixel adalah nilai pixel
                    b = b # b sama dengan nilai pixel

                self.Image.itemset((i, j), b) #mengubah nilai pixel gambar menjadi nilai pixel yang baru diubah kecerahannya
        self.displayImage(2)

    def ssim100(self): #fungsi ssim 100%
        sample = cv2.imread('cepe.jpg') #membaca gambar
        test = self.Image #membaca gambar

        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) #mengubah citra ke grayscale


        (score, diff) = structural_similarity(sample_gray, test, full=True) #menghitung nilai ssim
        print("SSIM score :", score) #menampilkan nilai ssim

        diff = (diff * 255).astype("uint8") #mengubah nilai diff ke uint8

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #mengubah nilai diff ke binary
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #mencari kontur
        contours = contours[0] if len(contours) == 2 else contours[1] #mencari kontur

        mask = np.zeros(sample.shape, dtype='uint8') #membuat matriks kosong dengan ukuran tinggi dan lebar citra
        filled_after = test.copy() #membuat matriks kosong dengan ukuran tinggi dan lebar citra

        for c in contours: #mengatur pergerakan kontur
            area = cv2.contourArea(c) #menghitung luas kontur
            if area > 40: #jika luas kontur lebih dari 40
                x, y, w, h = cv2.boundingRect(c)    #menghitung luas kotak
                cv2.rectangle(sample, (x, y), (x + w, y + h), (36, 255, 12), 2) #membuat kotak
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2) #membuat kotak
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1) #membuat kontur
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1) #membuat kontur

        cv2.imshow('sample', sample_gray) #menampilkan gambar
        cv2.imshow('citrapengujian', test) #menampilkan gambar
        cv2.imshow('perbedaan menggunakan negative filter', diff) #menampilkan gambar
        cv2.imshow("mask", mask) #menampilkan gambar
        cv2.imshow("filled after",filled_after) #menampilkan gambar
        self.image = filled_after #menampilkan gambar
        self.displayImage(2) #menampilkan gambar

        self.label_4.setText(str("SSIM Score = " + str(score))) #menampilkan nilai ssim

        if score < 0.97: #jika nilai ssim kurang dari 0.97
            print("Uang yang diinputkan adalah palsu") #menampilkan teks
            self.label_9.setText(str('Uang yang diinputkan adalah palsu')) #menampilkan teks
        else: #jika nilai ssim lebih dari 0.97
            print("Uang yang diinputkan adalah asli") #menampilkan teks
            self.label_9.setText(str('Uang yang diinputkan adalah asli')) #menampilkan teks

        cv2.imshow("Sample", sample) #menampilkan gambar
        cv2.imshow("Test", test) #menampilkan gambar
        cv2.imshow("Difference using negative filter", diff) #menampilkan gambar

        cv2.waitKey(0) #menunggu inputan dari keyboard


    def ssim50(self): #fungsi ssim 50%
        sample = cv2.imread('gocap.jpg') #membaca gambar
        test = self.Image #membaca gambar

        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) #mengubah citra ke grayscale


        (score, diff) = structural_similarity(sample_gray, test, full=True) #menghitung nilai ssim
        print("SSIM score :", score) #menampilkan nilai ssim

        diff = (diff * 255).astype("uint8") #mengubah nilai diff ke uint8

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #mengubah nilai diff ke binary
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #mencari kontur
        contours = contours[0] if len(contours) == 2 else contours[1] #mencari kontur

        mask = np.zeros(sample.shape, dtype='uint8') #membuat matriks kosong dengan ukuran tinggi dan lebar citra
        filled_after = test.copy() #membuat matriks kosong dengan ukuran tinggi dan lebar citra

        for c in contours: #mengatur pergerakan kontur
            area = cv2.contourArea(c) #menghitung luas kontur
            if area > 40: #jika luas kontur lebih dari 40
                x, y, w, h = cv2.boundingRect(c)   #menghitung luas kotak
                cv2.rectangle(sample, (x, y), (x + w, y + h), (36, 255, 12), 2) #membuat kotak
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2) #membuat kotak
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1) #membuat kontur
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1) #membuat kontur

        cv2.imshow('sample', sample_gray)   #menampilkan gambar
        cv2.imshow('citrapengujian', test) #menampilkan gambar
        cv2.imshow('perbedaan menggunakan negative filter', diff) #menampilkan gambar
        cv2.imshow("mask", mask) #menampilkan gambar
        cv2.imshow("filled after",filled_after) #menampilkan gambar
        self.image = filled_after #menampilkan gambar
        self.displayImage(2) #menampilkan gambar

        self.label_4.setText(str("SSIM Score = " + str(score))) #menampilkan nilai ssim

        if score < 0.97: #jika nilai ssim kurang dari 0.97
            print("Uang yang diinputkan adalah palsu") #menampilkan teks
            self.label_9.setText(str('Uang yang diinputkan adalah palsu')) #menampilkan teks
        else: #jika nilai ssim lebih dari 0.97
            print("Uang yang diinputkan adalah asli") #menampilkan teks
            self.label_9.setText(str('Uang yang diinputkan adalah asli')) #menampilkan teks

        cv2.imshow("Sample", sample) #menampilkan gambar
        cv2.imshow("Test", test) #menampilkan gambar
        cv2.imshow("Difference using negative filter", diff) #menampilkan gambar

        cv2.waitKey(0) #menunggu inputan dari keyboard

    def ssim100baru(self): #fungsi ssim 100%
        sample = cv2.imread('100rbbaru.jpg') #membaca gambar
        test = self.Image #membaca gambar

        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) #mengubah citra ke grayscale


        (score, diff) = structural_similarity(sample_gray, test, full=True) #menghitung nilai ssim
        print("SSIM score :", score) #menampilkan nilai ssim

        diff = (diff * 255).astype("uint8") #mengubah nilai diff ke uint8

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #mengubah nilai diff ke binary
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #mencari kontur
        contours = contours[0] if len(contours) == 2 else contours[1] #mencari kontur

        mask = np.zeros(sample.shape, dtype='uint8') #membuat matriks kosong dengan ukuran tinggi dan lebar citra
        filled_after = test.copy() #membuat matriks kosong dengan ukuran tinggi dan lebar citra

        for c in contours: #mengatur pergerakan kontur
            area = cv2.contourArea(c) #menghitung luas kontur
            if area > 40: #jika luas kontur lebih dari 40
                x, y, w, h = cv2.boundingRect(c)    #menghitung luas kotak
                cv2.rectangle(sample, (x, y), (x + w, y + h), (36, 255, 12), 2) #membuat kotak
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2) #membuat kotak
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1) #membuat kontur
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1) #membuat kontur

        cv2.imshow('sample', sample_gray) #menampilkan gambar
        cv2.imshow('citrapengujian', test) #menampilkan gambar
        cv2.imshow('perbedaan menggunakan negative filter', diff) #menampilkan gambar
        cv2.imshow("mask", mask) #menampilkan gambar
        cv2.imshow("filled after",filled_after) #menampilkan gambar
        self.image = filled_after #menampilkan gambar
        self.displayImage(2) #menampilkan gambar

        self.label_4.setText(str("SSIM Score = " + str(score))) #menampilkan nilai ssim

        if score < 0.97: #jika nilai ssim kurang dari 0.97
            print("Uang yang diinputkan adalah palsu") #menampilkan teks
            self.label_9.setText(str('Uang yang diinputkan adalah palsu')) #menampilkan teks
        else: #jika nilai ssim lebih dari 0.97
            print("Uang yang diinputkan adalah asli") #menampilkan teks
            self.label_9.setText(str('Uang yang diinputkan adalah asli')) #menampilkan teks

        cv2.imshow("Sample", sample) #menampilkan gambar
        cv2.imshow("Test", test) #menampilkan gambar
        cv2.imshow("Difference using negative filter", diff) #menampilkan gambar

        cv2.waitKey(0) #menunggu inputan dari keyboard

    def ssim50baru(self): #fungsi ssim 50%
        sample = cv2.imread('50rbbaru.jpg') #membaca gambar
        test = self.Image #membaca gambar

        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) #mengubah citra ke grayscale


        (score, diff) = structural_similarity(sample_gray, test, full=True) #menghitung nilai ssim
        print("SSIM score :", score) #menampilkan nilai ssim

        diff = (diff * 255).astype("uint8") #mengubah nilai diff ke uint8

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #mengubah nilai diff ke binary
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #mencari kontur
        contours = contours[0] if len(contours) == 2 else contours[1] #mencari kontur

        mask = np.zeros(sample.shape, dtype='uint8') #membuat matriks kosong dengan ukuran tinggi dan lebar citra
        filled_after = test.copy() #membuat matriks kosong dengan ukuran tinggi dan lebar citra

        for c in contours: #mengatur pergerakan kontur
            area = cv2.contourArea(c) #menghitung luas kontur
            if area > 40: #jika luas kontur lebih dari 40
                x, y, w, h = cv2.boundingRect(c)   #menghitung luas kotak
                cv2.rectangle(sample, (x, y), (x + w, y + h), (36, 255, 12), 2) #membuat kotak
                cv2.rectangle(test, (x, y), (x + w, y + h), (36, 255, 12), 2) #membuat kotak
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1) #membuat kontur
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1) #membuat kontur

        cv2.imshow('sample', sample_gray)   #menampilkan gambar
        cv2.imshow('citrapengujian', test) #menampilkan gambar
        cv2.imshow('perbedaan menggunakan negative filter', diff) #menampilkan gambar
        cv2.imshow("mask", mask) #menampilkan gambar
        cv2.imshow("filled after",filled_after) #menampilkan gambar
        self.image = filled_after #menampilkan gambar
        self.displayImage(2) #menampilkan gambar

        self.label_4.setText(str("SSIM Score = " + str(score))) #menampilkan nilai ssim

        if score < 0.97: #jika nilai ssim kurang dari 0.97
            print("Uang yang diinputkan adalah palsu") #menampilkan teks
            self.label_9.setText(str('Uang yang diinputkan adalah palsu')) #menampilkan teks
        else: #jika nilai ssim lebih dari 0.97
            print("Uang yang diinputkan adalah asli") #menampilkan teks
            self.label_9.setText(str('Uang yang diinputkan adalah asli')) #menampilkan teks


        cv2.waitKey(0) #menunggu inputan dari keyboard

    def displayImage(self, windows):    #fungsi untuk menampilkan gambar
        qformat = QImage.Format_Indexed8 #mengatur format gambar
        if len(self.Image.shape)==3: #jika ukuran gambar 3
            if(self.Image.shape[2])==4: #jika ukuran gambar 4
                qformat = QImage.Format_RGBA8888 #mengatur format gambar
            else: #jika ukuran gambar bukan 4
                qformat = QImage.Format_RGB888 #mengatur format gambar
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat) #mengatur ukuran gambar
        img = img.rgbSwapped() #mengatur ukuran gambar

        if windows == 1: #jika windows 1
            self.label_6.setPixmap(QPixmap.fromImage(img)) #menampilkan gambar
            self.label_6.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) #menampilkan gambar
            self.label_6.setScaledContents(True) #menampilkan gambar

        if windows == 2: #jika windows 2
            self.label_7.setPixmap(QPixmap.fromImage(img))
            self.label_7.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_7.setScaledContents(True)

if __name__ == "__main__": #fungsi main
    app = QtWidgets.QApplication(sys.argv) #membuat aplikasi
    window = ShowImage() #membuat objek
    window.show() #menampilkan objek
    sys.exit(app.exec_()) #menutup aplikasi
