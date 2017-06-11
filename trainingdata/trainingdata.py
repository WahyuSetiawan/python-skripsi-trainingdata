import sys
import os
import re
import subprocess

from Module import *
from Module import ProsesTrainingData
from view import MenuUtama
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    #'''
    app = QApplication(sys.argv)
    ex = MenuUtama.App()
    sys.exit(app.exec_())
    #'''

    '''
    trainingdata = ProsesTrainingData.TrainingData('data/sampleTweets.csv', 'data/feature_list/id-stopwords.txt', None)
    trainingdata.run()
    #'''
