# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import multiprocessing

import threading
 
from Module import ProsesTrainingData

class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.center()
        self.setFixedSize(700 ,700)  
      
        self.listview = QListWidget(self)
        self.listview.resize(700 - 20, 700 - 20 - 60)
        self.listview.move(10,10)
 
        button = QPushButton('Import File', self)
        button.setToolTip('This is an example button')
        button.resize(200, 40)
        button.move(10,700 - 10 - 40) 
        button.clicked.connect(self.on_click)

        button_run = QPushButton('Run', self)
        button_run.setToolTip('Menjalankan program training data')
        button_run.resize(200, 40)
        button_run.move(700 - 200 - 10, 700 - 10 - 40) 
        button_run.clicked.connect(self.on_click_run)
    
        self.show()

    def center(self):
        #self.setGeometry(self.left, self.top, self.width, self.height)
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    @pyqtSlot()
    def on_click(self):
        filename = self.openFileNameDialog()
        stopword = self.openFileStopword()

        if filename: 
            self.trainingdata = ProsesTrainingData.TrainingData(filename, stopword, self.listview)
            self.listview.addItem("sukses masukan data")

    @pyqtSlot()
    def on_click_run(self):
        #self.trainingdata.run()
        return
 
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Import File Dataset", "","All Files (*);;Python Files (*.sav)", options=options)
        if fileName:
            return fileName

    def openFileStopword(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Import File Stopwords", "","All Files (*);;Python Files (*.txt)", options=options)
        if fileName:
            return fileName


