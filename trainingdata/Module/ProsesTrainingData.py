import re
import sys
import csv
import math
import os
import pickle
import copy
import multiprocessing as Pool
import warnings

from PyQt5.QtWidgets import QListWidget
from PyQt5.QtCore import pyqtSignal

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from Utility import tfidf
from Utility import Particle

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

from pyswarm import pso

from matplotlib import style
style.use("ggplot")

positive = 'positive'
negative = 'negative'
neutral = 'neutral'

stopwords = None

featurelist = {
            positive : [], 
            negative : [],
            neutral : []
            }
document = {
            positive : [], 
            negative : [],
            neutral : []
            }
tfidfresult = {
            positive : [], 
            negative : [],
            neutral : []
            }

tfidfweight = {
            positive : [], 
            negative : [],
            neutral : []
            }

particle = {
            positive : [], 
            negative : [],
            neutral : []
            }

x1 = []
x2 = []

label = []

clf = None

class TrainingData:
    update = pyqtSignal(str)

    def __init__(self, tweet, stopword, qlistwidget = None):
       self.st = open(stopword, 'r')
       self.stopWords = self.getStopWordList(stopword)

       #Read the tweets one by one and process it
       ##inpTweets = csv.reader(open('data/sampleTweets.csv', 'rb'), delimiter=',', quotechar='|')

       csvfile = open(tweet, "r")
       self.inpTweets = csv.reader(csvfile)

       self.list = qlistwidget

       warnings.filterwarnings("ignore", category=DeprecationWarning)

    #start getStopWordList
    def getStopWordList(self,stopWordListFileName):
        #read the stopwords file and build a list
        stopWords = []
        stopWords.append('AT_USER')
        stopWords.append('URL')

        fp = open(stopWordListFileName, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopWords.append(word)
            line = fp.readline()
        fp.close()
        return stopWords
    #end

    #memulai filter tweet
    def processTweet(self,tweet):
        # process the tweets

        # menganti kata kapital menjadi kecil
        tweet = tweet.lower()
    
        # menganti kata https://* dan www.* menjadi url
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)

        # menganti @usename ke AT_USER
        # tweet = re.sub('@[^\s]+','AT_USER',tweet)

        # menghilangkan spasi
        tweet = re.sub('[\s]+', ' ', tweet)

        # menghilangkan hastag
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

        # potong
        tweet = tweet.strip('\'"')

        # menghilangkan kata kata unicode untuk emoticon
        tweet = tweet.encode('ascii', 'ignore').decode('unicode_escape')

        tweet =  re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)

        myre = re.compile(u'['
                    u'\U0001F300-\U0001F5FF'
                    u'\U0001F600-\U0001F64F'
                    u'\U0001F680-\U0001F6FF'
                    u'\u2600-\u26FF\u2700-\u27BF]+', 
                    re.UNICODE)

        # menghilangkan unicode
        tweet = myre.sub('', tweet)

        # menghilangkan username
        tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    
        return tweet
    #akhir

    #emlalkukan preprocessing t4erhadap data yang digunakan
    def preprocessingData(self,tweet, stopwords):
        # mempersiapkan varible pembantu
        featureList = []
        tmpFeature = []

        # mempersiapkan api untuk menjadalankan preprocessing dari tokenizer dan steamming
        nltktokenizer = TweetTokenizer()
        factorysteammer = StemmerFactory()
        stemmer = factorysteammer.create_stemmer()

        #proses tokenizer
        featureList = nltktokenizer.tokenize(tweet)

        # menghilangkan kata stopwords
        for w in featureList:
            if w not in stopwords:
                tmpFeature.append(w)
    
        featureList = tmpFeature
        tmpFeature = []

        # proses steamming perkata agar mendapatkan kata baku
        for w in featureList:
            tmpFeature.append(stemmer.stem(w))

        featureList = tmpFeature
        tmpFeature = []

        return featureList

    # vectorizer
    def TFIDF(self,tweet): 
        vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)

        tmpTfidf = vectorizer.fit_transform(tweet)

        return tmpTfidf

    #persamaan svm untuk melakukan predik
    def persamaanSVM(self, X):
        return self.clf.predict_log_proba(X)

    #menyimpan mode4l presintence dari svm untuk dilakukan testing di website
    def exportModel(self,filename, featurelist):
        with open('featurelist.txt', 'a') as f:
            
            for i, feature in enumerate(featurelist):
                self.cetak(feature)
                for a in featurelist[feature]:
                    self.cetak(a)
                    f.write(''.join([a,",", feature]))
                    f.write("\n")
                
        pickle.dump(self.svm, open(filename, 'wb'))

    #function untuk menjalankan pso
    def runpso(self, particles,label):
        swarm = {
            positive : [], 
            negative : [],
            neutral : []
            }

        temp = []

        #preparing data for pso particle, append in one variable particle pso
        for index, value in enumerate(particles):
            partic = Particle.Particle(particles[index], index, self.clf, [i for i, no in enumerate(self.clf.classes_) if no == label[index]], self.clf.class_weight_)
            swarm[label[index]].append(partic)
            temp.append(partic)

        for step in range(20):
            print("---Itteration " + str(step) + "---")

            hasilkedua = self.pso(swarm)

            svmPertama = copy.deepcopy(self.svmStandard)
            svmPertama.fit(self.X, self.y)

            svmKedua = copy.deepcopy(self.svmStandard)
            svmKedua.fit(hasilkedua, self.y)

            #perbandingan akurasi pada kedua svm memiliki akurasi dibandingkan sebelumnya atau tidak
            if svmKedua.score(self.x1, self.y) > svmPertama.score(self.x1,self.y):
                self.X = copy.deepcopy( hasilkedua)        

    def pso(self, swarm):
        #do pso litetation 
        temp = copy.deepcopy(self.X)

        # perubahan particle berdasarkan feature mereka
        for indexswarm, valueswarm in enumerate(swarm):
            #persiapan variable yang digunakan untuk keperluan pso
            swarm[valueswarm] = sorted(swarm[valueswarm])
            indexvalue = [i for i, no in enumerate(self.clf.classes_) if no == valueswarm]
            weight = self.clf.class_weight_[indexvalue][0]

            gbest = 0

            #penentuan gbest bedasarkan hasil dari akumulasi persamaan svm
            for indexpbest, particlepbest in enumerate( swarm[valueswarm]):
                pbest = self.clf.decision_function(particlepbest.position)[0][indexvalue]
                if pbest >= gbest :
                    gbest = pbest
                    globalBest = swarm[valueswarm][indexpbest]  
                    self.solution = globalBest

            #perubahan pada particle swarm
            for particle in swarm[valueswarm]:
                particle.updateParticle(globalBest.getPositionList(), weight)
                
            swarm[valueswarm] = sorted(swarm[valueswarm])
            
            #Print swarm
            for particle in swarm[valueswarm]:     
                temp[particle.index] = particle.position

        return temp

    # run program pemanggilan data
    def run(self):
        global positive
        global negative
        global neutral

        tweets = []

        self.x1 = []
        self.svmStandard = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape="ovr", degree=3, gamma='auto', kernel='linear',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

        self.tfidfDocument = tfidf.TfIdf()

        # start loop
        for i, row in enumerate(self.inpTweets):
            sentiment = row[0].replace('|','')
            tweet = row[1].replace('|', '')

            self.cetak(''.join(["preprocessing data ke ", str(i)," tweet : ", tweet]))

            #tahap preprocessing
            processedTweet = self.processTweet(tweet)
            featureVector = self.preprocessingData(processedTweet, self.stopWords)
            tweets.append((featureVector, sentiment))

             # tahap binary
            self.tfidfDocument.add_document(i,featureVector)

            if (sentiment == positive ):
                document[positive].append(tweet)

                for feature in featureVector:
                    featurelist[positive].append(feature)

                featurelist[positive] = list(set(featurelist[positive]))

                label.append(positive)

            if (sentiment == negative ):
                document[negative].append(tweet)

                for feature in featureVector:
                   featurelist[negative].append(feature)

                featurelist[negative] = list(set(featurelist[negative]))

                label.append(negative)

            if (sentiment == neutral ):
                document[neutral].append(tweet)

                for feature in featureVector:
                   featurelist[neutral].append(feature)

                featurelist[neutral] = list(set(featurelist[neutral]))

                label.append(neutral)
        
        # mendapatkan pembobotan menggunakan tf idf
        for i, feature in enumerate(featurelist):
            #self.cetak("generating tf idf per feature : ".join(feature))
            tfidfresult[feature] = self.tfidfDocument.similarities(featurelist[feature])
            
            for x in tfidfresult[feature]:
                tfidfweight[feature].append(x[1])

            print(tfidfresult[feature])
            print(tfidfweight[feature])
     
        # merubah ke variable yang bisa diterima oleh svm
        for i, row in enumerate(tfidfweight[positive]):
            a = [tfidfresult[positive][i][1],tfidfresult[negative][i][1], tfidfresult[neutral][i][1]]
            self.x1.append(a)
            print(a)

        #print(self.x1)
        self.cetak('perhitungan svm dilanjutkan dengan perhitungan pso')

        self.X = copy.deepcopy(self.x1)
        self.y = label
        
        self.clf =  svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape="ovr", degree=3, gamma='auto', kernel='linear',
            max_iter=-1, probability=True, random_state=None, shrinking=True,
            tol=0.001, verbose=True)
        #self.clf = copy.deepcopy(self.svmStandard)
        self.clf.fit(self.X,self.y)

        #run pso
        self.runpso(self.x1, self.y)

        self.svm = copy.deepcopy(self.svmStandard)
        self.svm.fit(self.X, self.y)
        asss = self.clf.decision_function(self.x1)

        print(asss.shape)
        print(self.clf.class_weight_)
        print(self.clf.predict(self.x1))
        print(self.clf.support_ )
        print(self.clf.score(self.x1, self.y))
        print(self.clf.classes_)
        print(self.clf.support_vectors_)
        print("cofisien for svm :")
        print(self.clf.coef_)
        print("dual cofisien for svm :")
        print(self.clf.dual_coef_)
        
        
        print(self.svm.decision_function(self.x1))
        print(self.svm.predict(self.x1))
        print(self.svm.support_vectors_ )
        print(self.svm.score(self.x1, self.y))

        self.exportModel('modelterbaru.pkl', featurelist)

        self.cetak(''.join(['Selesai training model disimpan dalam ', os.path.dirname(__file__) , 'modelterbaru.pkl']))

        '''
        a = x1
        print(clf.classes_)

        dicision = self.persamaanSVM(a)
        #dicision = clf.predict_proba(a)
        print(dicision)
        print(clf.predict(a))

        #print(clf.predict([10.58,10.76,0.04013377926421405]))
        #print(clf.predict([0, 0.09126984126984126,  1.3140096618357489]))
        #print(clf.get_params())
        w = clf.coef_[0]
        #print(w)

        a = -w[0] / w[1]

        xx = np.linspace(0,12)
        yy = a * xx - clf.intercept_[0] / w[1]

        h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

        #'''

    def cetak(self, pesan):
        print(pesan)
        self.update.emit(pesan)
'''
        plt.scatter(X[:, 0], X[:, 1], c = y)
        plt.legend()
        plt.show()



'''
