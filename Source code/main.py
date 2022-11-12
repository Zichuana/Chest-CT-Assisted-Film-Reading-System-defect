import sys

from PyQt5.QtWidgets import QApplication


from DrawCradCam import DrawCradCams
import os
from predict import Predict


from PyQt5 import QtWidgets, QtGui, QtCore

import image as PyUI

x = 0
y = 0
z = 0
a = 0
n = 0
res0 = []
s = 1
we = None
flag = 0

class RunWindow(PyUI.Ui_MainWindow):

    def __init__(self, MainWindow):
        super(RunWindow, self).__init__()
        self.setupUi(MainWindow)
        self.MainWindow = MainWindow
        self.MainWindow.show()
        self.RunFunc()
        self.species = {
            0: ["结果类型"], 1: ["negative", "positive"], 2: ["COVID-19", "Normal", "Pneumonia"]
        }
        self.res1 = []
        self.res2 = []
        self.res3 = []
        self.dir_2c = {}

    def BatchPredict_3(self, pic_path):
        img_data = []
        pic_path = pic_path
        filenames = os.listdir(pic_path)
        for filename in filenames:
            img_data.append(pic_path+'/'+filename)
            print(pic_path+filename)
        # print(img_data)
        dif = 0
        for i in img_data:
            dif += 1
            self.textEdit.append("PROCESSIONG NO.{} CT:{}".format(dif, i))
            QApplication.processEvents()
            pre = Predict(i)
            draw = DrawCradCams(i, dif)
            pro_res, result = pre.predict_3()
            # print(pro_res)
            if pro_res.find("COVID-19") > 0:
                self.res1.append([i, pro_res, result])  # 0-路径 1-最可能的结果 2-所有结果
            if pro_res.find("Normal") > 0:
                self.res2.append([i, pro_res, result])
            if pro_res.find("Pneumonia") > 0:
                self.res3.append([i, pro_res, result])
            save_path = draw.DRAW()
            self.dir_2c[i] = save_path
            print("FINISH {}".format(dif))
            self.textEdit.append("FINISH".format(dif))
            QApplication.processEvents()
        self.textEdit.clear()

    def BatchPredict_2(self, pic_path):
        img_data = []
        pic_path = pic_path
        filenames = os.listdir(pic_path)
        for filename in filenames:
            img_data.append(pic_path+'/'+filename)
            print(pic_path+filename)
        # print(img_data)
        dif = 0
        for i in img_data:
            dif += 1
            self.textEdit.append("PROCESSIONG NO.{} X-RAY:{}".format(dif, i))
            QApplication.processEvents()
            pre = Predict(i)
            draw = DrawCradCams(i, dif)
            pro_res, result = pre.predict_2()
            if pro_res.find("negative") > 0:
                self.res1.append([i, pro_res, result])  # 0-路径 1-最可能的结果 2-所有结果
            if pro_res.find("positive") > 0:
                self.res2.append([i, pro_res, result])
            # .append([i, pro_res, result])
            save_path = draw.DRAW()
            self.dir_2c[i] = save_path
            print("**FINISH {}**".format(dif))
            self.textEdit.append("FINISH".format(dif))
            QApplication.processEvents()
            # print(self.res1)
            # print(self.res2)
        self.textEdit.clear()

    def SelectImage(self):
        ImagePath, ok = QtWidgets.QFileDialog.getOpenFileName(self.MainWindow, '打开文件', '.',
                                                              filter='图像文件(*.png *.jpg);;All Files (*);;')
        if ok:
            img = QtGui.QPixmap(ImagePath)
            img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.label_image.setPixmap(img)  # 显示图片
            self.lineEdit.setText(ImagePath)  # 显示图片文件路径
            return ImagePath
        else:
            return

    def Selectfolder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹",
                                                               r"D:\2022.4\19project\CTorXCOVID19NetProject")  # 起始路径
        if directory != '':
            self.lineEdit.setText(directory)
            return directory
        else:
            return

    def SelectParse(self):
        self.res1.clear()
        self.res2.clear()
        self.res3.clear()
        if x != 0 and y != 0:
            self.textEdit.clear()
            self.label_image.setPixmap(QtGui.QPixmap(""))
            self.label_ShowImage.setPixmap(QtGui.QPixmap(""))
        global a, n, res0, s, we
        if y == 1:
            if x == 1:
                we = None
                we = self.SelectImage()
                if we == None:
                    return
                img_path = we
                pre = Predict(img_path)
                self.textEdit.append("Image processing, please wait...")
                QApplication.processEvents()
                pro_res, result = pre.predict_2()
                self.textEdit.clear()
                self.textEdit.append(pro_res)
                for i in result:  # 预测的结果
                    self.textEdit.append(i)
                draw = DrawCradCams(img_path, 0)  # 定义绘制热力图对象
                save_path = draw.DRAW()  # 分析出来的图片的路径
                img = QtGui.QPixmap(save_path)
                img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                self.label_ShowImage.setPixmap(img)  # 显示图片
                os.remove(save_path)  # 此分析结束的时候 删除分析出来的热力图（该热力图可以类比于缓存）
            elif x == 2:
                we = None
                we = self.SelectImage()
                if we == None:
                    return
                img_path = we
                pre = Predict(img_path)
                self.textEdit.append("Image processing, please wait...")
                QApplication.processEvents()
                pro_res, result = pre.predict_3()
                self.textEdit.clear()
                self.textEdit.append(pro_res)
                for i in result:  # 预测的结果
                    self.textEdit.append(i)
                draw = DrawCradCams(img_path, 0)  # 定义绘制热力图对象
                save_path = draw.DRAW()  # 分析出来的图片的路径
                img = QtGui.QPixmap(save_path)
                img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                self.label_ShowImage.setPixmap(img)
                os.remove(save_path)  # 此分析结束的时候 删除分析出来的热力图（该热力图可以类比于缓存）
        elif y == 2:
            we = None
            we = self.Selectfolder()
            if we == None:
                return
            pic_path = we
            if x == 1:
                for i in os.listdir("./2CData"):
                    os.remove('./2CData/' + i)
                self.BatchPredict_2(pic_path)
                QApplication.processEvents()
                if z == 0:
                    n = len(self.res1)
                    if n == 0:
                        self.textEdit.append("该类别中无图像！")
                        return
                    res0 = self.res1
                    s = 0
                    i = self.res1[0]
                    self.textEdit.append(i[1])
                    img = QtGui.QPixmap(i[0])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_image.setPixmap(img)
                    img = QtGui.QPixmap(self.dir_2c[i[0]])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_ShowImage.setPixmap(img)
                    # print(1234567890)
                    for j in i[2]:
                        self.textEdit.append(j)
                if z == 1:
                    n = len(self.res2)
                    if n == 0:
                        self.textEdit.append("该类别中无图像！")
                        return
                    s = 0
                    res0 = self.res2
                    i = self.res2[0]
                    self.textEdit.append(i[1])
                    img = QtGui.QPixmap(i[0])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_image.setPixmap(img)
                    img = QtGui.QPixmap(self.dir_2c[i[0]])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_ShowImage.setPixmap(img)
                    # print(12345678900000000)
                    for j in i[2]:
                        self.textEdit.append(j)
            elif x == 2:
                for i in os.listdir("./2CData"):
                    os.remove('./2CData/' + i)
                self.BatchPredict_3(pic_path)
                QApplication.processEvents()
                if z == 0:
                    # dir_2c[res[][0]]是res[][0]对应的热力图
                    n = len(self.res1)
                    if n == 0:
                        self.textEdit.append("该类别中无图像！")
                        return
                    res0 = self.res1
                    s = 0
                    i = self.res1[0]
                    self.textEdit.append(i[1])
                    img = QtGui.QPixmap(i[0])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_image.setPixmap(img)
                    img = QtGui.QPixmap(self.dir_2c[i[0]])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_ShowImage.setPixmap(img)
                    for j in i[2]:
                        self.textEdit.append(j)
                if z == 1:
                    n = len(self.res2)
                    if n == 0:
                        self.textEdit.append("该类别中无图像！")
                        return
                    s = 0
                    res0 = self.res2
                    i = self.res2[0]
                    self.textEdit.append(i[1])
                    img = QtGui.QPixmap(i[0])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_image.setPixmap(img)
                    img = QtGui.QPixmap(self.dir_2c[i[0]])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_ShowImage.setPixmap(img)
                    for j in i[2]:
                        self.textEdit.append(j)
                if z == 2:
                    n = len(self.res3)
                    if n == 0:
                        self.textEdit.append("该类别中无图像！")
                        return
                    s = 0
                    res0 = self.res3
                    i = self.res3[0]
                    self.textEdit.append(i[1])
                    img = QtGui.QPixmap(i[0])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_image.setPixmap(img)
                    img = QtGui.QPixmap(self.dir_2c[i[0]])
                    img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
                    self.label_ShowImage.setPixmap(img)
                    for j in i[2]:
                        self.textEdit.append(j)
        pass

    def up(self):
        global n, res0, s
        if y == 2:
            if n == 0:
                self.textEdit.append("该类别中无图像！")
                return
            s -= 1
            if s < 0:
                if s < -1:
                    s += 1
                self.textEdit.clear()
                self.textEdit.setText("上一张没有图像了！")
                self.label_image.setPixmap(QtGui.QPixmap(""))
                self.label_ShowImage.setPixmap(QtGui.QPixmap(""))
                return
            i = res0[s]
            self.textEdit.clear()
            self.textEdit.append(i[1])
            self.label_image.setPixmap(QtGui.QPixmap(""))
            self.label_ShowImage.setPixmap(QtGui.QPixmap(""))
            img = QtGui.QPixmap(i[0])
            img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.label_image.setPixmap(img)
            img = QtGui.QPixmap(self.dir_2c[i[0]])
            img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.label_ShowImage.setPixmap(img)
            for j in i[2]:
                self.textEdit.append(j)
        pass

    def down(self):
        global n, res0, s
        # print(n)
        # print('a', s)
        if y == 2:
            if n == 0:
                self.textEdit.append("该类别中无图像！")
                return
            # print('b', s)
            s += 1
            # print('1', s)
            if s >= n:
                if s > n:
                    s -= 1
                # print('2', s)
                self.textEdit.clear()
                self.textEdit.setText("已全部展示完！")
                self.label_image.setPixmap(QtGui.QPixmap(""))
                self.label_ShowImage.setPixmap(QtGui.QPixmap(""))
                return
            i = res0[s]
            self.textEdit.clear()
            self.textEdit.append(i[1])
            self.label_image.setPixmap(QtGui.QPixmap(""))
            self.label_ShowImage.setPixmap(QtGui.QPixmap(""))
            img = QtGui.QPixmap(i[0])
            img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.label_image.setPixmap(img)
            img = QtGui.QPixmap(self.dir_2c[i[0]])
            img = img.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.label_ShowImage.setPixmap(img)
            for j in i[2]:
                self.textEdit.append(j)
        pass

    def baocun2(self):
        global y, x
        y = self.comboBox_2.currentIndex()
        if y == 1:
            self.comboBox_3.clear()
        if y == 2:
            self.comboBox_3.clear()
            self.comboBox_3.addItems(self.species[x])
        pass

    def baocun3(self, t):
        global z
        z = t
        pass

    def changed(self, text):
        global x
        x = text
        sp = self.species[text]
        self.comboBox_3.clear()
        self.comboBox_3.addItems(sp)

    def RunFunc(self):
        self.comboBox_2.currentIndexChanged.connect(self.baocun2)
        self.comboBox_3.currentIndexChanged.connect(self.baocun3)
        self.comboBox.activated.connect(self.changed)
        self.pushButton_cmd.clicked.connect(self.SelectParse)
        self.pushButton_down.clicked.connect(self.down)
        self.pushButton_up.clicked.connect(self.up)
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = RunWindow(MainWindow)
    sys.exit(app.exec_())

