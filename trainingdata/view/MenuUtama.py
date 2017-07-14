# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal
import multiprocessing

import threading
 
from Module import ProsesTrainingData

class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'Training data Twitter'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.center()
        self.setFixedSize(700 ,700)  
 
        button = QPushButton('Import File', self)
        button.setToolTip('Untuk memilih file yang akan dipilih')
        button.resize(200, 40)
        button.move(10, self.frameGeometry().height() - button.frameGeometry().height() - 10) 
        button.clicked.connect(self.on_click)

        button_run = QPushButton('Run', self)
        button_run.setToolTip('Menjalankan program training data')
        button_run.resize(button.frameGeometry().width(), button.frameGeometry().height())
        button_run.move(self.frameGeometry().width() - button_run.frameGeometry().width() - 10, button.frameGeometry().y()) 
        button_run.clicked.connect(self.on_click_run)

        self.listview = QListWidget(self)
        self.listview.move(10,10)
        self.listview.resize(self.frameGeometry().width() - 20, button.frameGeometry().y() - self.listview.frameGeometry().y() - 10)
    
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
        QMessageBox.question(self, "Pesan", "Anda telah memilih file Training, sekarang anda memilih file stopword", QMessageBox.Yes)
        self.insertStrListView("".join(["Data training yang digunakan : ", filename]))

        stopword = self.openFileStopword()
        QMessageBox.question(self, "Pesan", "Anda telah memilih file Stopwords, sekarang data telah siap untuk ditraining", QMessageBox.Yes)
        self.insertStrListView("".join(["File stopword yang digunakan : ", stopword]))

        if filename: 
            self.threadTrainingData = ThreadTrainingData(filename, stopword)
            self.threadTrainingData.update.connect(self.insertStrListView)
            self.listview.addItem("Data training telah dimaukan, training data telah siap")

    @pyqtSlot()
    def on_click_run(self):
        self.threadTrainingData.start()
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

    def insertStrListView(self, message):
        self.listview.addItem(message)
        self.listview.scrollToBottom()
        return 

class ThreadTrainingData(QThread):
    update = pyqtSignal(str)
    finish = pyqtSignal()

    def __init__(self, filename, stopword):
        QThread.__init__(self)
        self.filename = filename
        self.stopword = stopword
        self.trainingdata = ProsesTrainingData.TrainingData(filename, stopword)
        self.trainingdata.update = self.update

    def __del__(self):
        self.wait()

    def run(self):

        if os.path.exists(self.filename) and os.path.exists(self.stopword) :
            self.trainingdata.run()
        else :
            self.update.emit("File training dan stopword tidak dapat ditemukan")
        #return super().run()