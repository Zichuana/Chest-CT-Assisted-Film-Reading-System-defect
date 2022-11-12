from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 1250)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("图标.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("#MainWindow{background-color:rgb(220, 220, 220)}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.horizontalLayout.addLayout(self.horizontalLayout_2)
        self.pushButton_cmd = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_cmd.setFont(font)
        self.pushButton_cmd.setObjectName("pushButton_cmd")
        self.horizontalLayout.addWidget(self.pushButton_cmd)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_image.sizePolicy().hasHeightForWidth())
        self.label_image.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_image.setFont(font)
        self.label_image.setObjectName("label_image")
        self.horizontalLayout_3.addWidget(self.label_image)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("图像种类")
        self.comboBox.addItems(["X-ray", "CT"])      # 字典键值

        self.verticalLayout.addWidget(self.comboBox)

        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItems(["选择类型", "单张", "多张"])

        self.verticalLayout.addWidget(self.comboBox_2)

        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.comboBox_3.setFont(font)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("分析类型")
        self.comboBox_3.addItem("COVID-19")
        self.comboBox_3.addItem("Normal")
        self.comboBox_3.addItem("Pneumonia")
        self.verticalLayout.addWidget(self.comboBox_3)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.label_ShowImage = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_ShowImage.sizePolicy().hasHeightForWidth())
        self.label_ShowImage.setSizePolicy(sizePolicy)
        self.label_ShowImage.setText("")
        self.label_ShowImage.setObjectName("label_ShowImage")
        self.horizontalLayout_3.addWidget(self.label_ShowImage)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_up = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_up.setFont(font)
        self.pushButton_up.setObjectName("pushButton_up")
        self.verticalLayout_2.addWidget(self.pushButton_up)
        self.pushButton_down = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_down.setFont(font)
        self.pushButton_down.setObjectName("pushButton_down")
        self.verticalLayout_2.addWidget(self.pushButton_down)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 2)
        self.horizontalLayout_3.setStretch(3, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_3.addWidget(self.textEdit)
        self.verticalLayout_3.setStretch(1, 7)
        self.verticalLayout_3.setStretch(3, 2)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(MainWindow)
        self.comboBox_3.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像分析"))
        self.label.setText(_translate("MainWindow", "所选路径："))
        self.pushButton_cmd.setText(_translate("MainWindow", "选择文件/文件夹"))
        self.label_image.setText(_translate("MainWindow", "请点击开始分析："))
        self.comboBox.setItemText(0, _translate("MainWindow", "图像种类"))
        self.comboBox.setItemText(1, _translate("MainWindow", "X-ray"))
        self.comboBox.setItemText(2, _translate("MainWindow", "CT"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "分析类型"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "单张"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "多张"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "结果类型"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "COVID-19"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "Normal"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "Pneumonia"))
        self.pushButton_up.setText(_translate("MainWindow", "上一张"))
        self.pushButton_down.setText(_translate("MainWindow", "下一张"))
        self.label_3.setText(_translate("MainWindow", "分析结果"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:14pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">请先选择图像种类、分析类型、结果类型(单张无需选择结果类型)</p>\n最后选择文件/文件夹路径...</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
