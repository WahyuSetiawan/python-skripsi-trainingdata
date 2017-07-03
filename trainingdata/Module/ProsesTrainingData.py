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
from Utility import tfidf, Particle

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
    
        # menhapus kata https://* dan www.*
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)

        # meghapus @usename
        tweet = re.sub('@[^\s]+','',tweet)

        # menghilangkan spasi
        tweet = re.sub('[\s]+', ' ', tweet)

        # menghilangkan hastag
        #tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        tweet = re.sub(r'#([^\s]+)', '', tweet)

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
        #tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    
        return tweet
    #akhir

    #emlalkukan preprocessing t4erhadap data yang digunakan
    def preprocessingData(self,tweet, stopwords):
        self.cetak("Tweet bersih : " + tweet)

        # mempersiapkan varible pembantu
        featureList = []
        tmpFeature = []

        # mempersiapkan api untuk menjadalankan preprocessing dari tokenizer dan steamming
        nltktokenizer = TweetTokenizer()
        factorysteammer = StemmerFactory()
        stemmer = factorysteammer.create_stemmer()

        #proses tokenizer
        self.cetak("Hasil Tokenization :")
        featureList = nltktokenizer.tokenize(tweet)
        self.cetak(''.join(str(item) + " " for item in featureList))

        # menghilangkan kata stopwords
        self.cetak("Hasil filter Stopwords :")
        for w in featureList:
            if w not in stopwords:
                tmpFeature.append(w)
    
        featureList = tmpFeature
        self.cetak(''.join(str(item) + " " for item in featureList))
        tmpFeature = []

        # proses steamming perkata agar mendapatkan kata baku
        self.cetak("Hasil Stemming perkata:")
        for w in featureList:
            tmpFeature.append(stemmer.stem(w))

        featureList = tmpFeature
        self.cetak(''.join(str(item) + " " for item in featureList))
        tmpFeature = []

        self.cetak("")

        return featureList

    # vectorizer
    def TFIDF(self,tweet): 
        vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)

        tmpTfidf = vectorizer.fit_transform(tweet)

        return tmpTfidf

    #persamaan svm untuk melakukan predik
    def persamaanSVM(self, X):
        return self.svmsebelumpso.predict_log_proba(X)

    #menyimpan mode4l presintence dari svm untuk dilakukan testing di website
    def exportModel(self,filename, featurelist):
        with open('featurelist.txt', 'a') as f:

            
            self.cetak("Mengeluarkan perhitungan dalam bentuk PKL :")
            pickle.dump(self.svm, open(filename, 'wb'))
            self.cetak(''.join(['Selesai training model disimpan dalam ', os.path.dirname(__file__) , 'modelterbaru.pkl']))
            
            self.cetak("Mengeluarkan Feature List dalam bentuk TXT :")

            for i, feature in enumerate(featurelist):
                self.cetak(feature.join(" :"))
                for a in featurelist[feature]:
                    self.cetak(a)
                    f.write(''.join([a,",", feature]))
                    f.write("\n")
                self.cetak("")
                self.cetak("")

            self.cetak(''.join(['Selesai Feature List disimpan dalam ', os.path.dirname(__file__) , 'featurelist.txt']))                


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
            indexClass = [i for i, no in enumerate(self.svmsebelumpso.classes_) if no == label[index]]

            print(self.svmsebelumpso.coef_[indexClass])

            partic = Particle.Particle(
                particles[index], 
                index, 
                self.svmsebelumpso, 
                [i for i, no in enumerate(self.svmsebelumpso.classes_) if no == label[index]],
                self.svmsebelumpso.coef_[indexClass][0]
            )

            swarm[label[index]].append(partic)
            temp.append(partic)

        for step in range(10):
            self.cetak("---Itteration " + str(step) + "---")

            hasilkedua = self.pso(swarm)

            svmPertama = copy.deepcopy(self.svmStandard)
            svmPertama.fit(self.X, self.y)

            svmKedua = copy.deepcopy(self.svmStandard)
            svmKedua.fit(hasilkedua, self.y)

            self.cetak(str(svmKedua.score(self.x1, self.y)) + ' > ' + str(svmPertama.score(self.x1,self.y)))

            #perbandingan akurasi pada kedua svm memiliki akurasi dibandingkan sebelumnya atau tidak
            if svmKedua.score(self.x1, self.y) > svmPertama.score(self.x1,self.y):
                self.cetak("Hasil Perbaikan Parameter diterima")
                self.X = copy.deepcopy( hasilkedua)        

    def pso(self, swarm):
        #do pso litetation 
        temp = copy.deepcopy(self.X)

        # perubahan particle berdasarkan feature mereka
        for indexswarm, valueswarm in enumerate(swarm):
            #persiapan variable yang digunakan untuk keperluan pso
            swarm[valueswarm] = sorted(swarm[valueswarm])
            indexvalue = [i for i, no in enumerate(self.svmsebelumpso.classes_) if no == valueswarm]
            weight = self.svmsebelumpso.class_weight_[indexvalue][0]

            gbest = 0

            globalBest = []

            #penentuan gbest bedasarkan hasil dari akumulasi persamaan svm
            for indexpbest, particlepbest in enumerate(swarm[valueswarm]):
                pbest = self.svmsebelumpso.decision_function(particlepbest.position)[0][indexvalue]
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

            self.cetak(''.join(["Preprocessing data ke ", str(i)," tweet : ", tweet]))

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

        self.cetak("")
        self.cetak("Perhitungan TF-IDF :")
     
        # merubah ke variable yang bisa diterima oleh svm
        for i, row in enumerate(tfidfweight[positive]):
            a = [tfidfresult[positive][i][1],tfidfresult[negative][i][1], tfidfresult[neutral][i][1]]
            self.x1.append(a)
            self.cetak("Hasil TF-IDF Tweet Ke - " + str(i))
            self.cetak("Hasil TF-IDF : " + str(a[0]) +", " + str(a[1]) + ", " + str(a[2]))

        #print(self.x1)
        self.cetak(" ")
        self.cetak('Perhitungan SVM :')

        self.X = copy.deepcopy(self.x1)
        self.y = label
        
        self.svmsebelumpso =  svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape="ovr", degree=3, gamma='auto', kernel='linear',
            max_iter=-1, probability=True, random_state=None, shrinking=True,
            tol=0.001, verbose=True)

        #self.svmsebelumpso = copy.deepcopy(self.svmStandard)
        self.svmsebelumpso.fit(self.X,self.y)
        asss = self.svmsebelumpso.decision_function(self.x1)
        print(asss.shape)
        print(self.svmsebelumpso.class_weight_)
        print(self.svmsebelumpso.predict(self.x1))
        print(self.svmsebelumpso.support_ )
        print(self.svmsebelumpso.score(self.x1, self.y))
        print(self.svmsebelumpso.classes_)
        print(self.svmsebelumpso.support_vectors_)
        print("cofisien for svm :")
        print(self.svmsebelumpso.coef_)
        print("dual cofisien for svm :")
        print(self.svmsebelumpso.dual_coef_)

        self.cetak(''.join(str(e) + " " for e in self.svmsebelumpso.predict(self.x1)))
        self.cetak(str(self.svmsebelumpso.score(self.x1, self.y)))
        
        self.cetak("")
        self.cetak("Pehitungan PSO terhadap Paramter SVM :")
        #run pso
        self.runpso(self.x1, self.y)

        self.svm = copy.deepcopy(self.svmStandard)
        self.svm.fit(self.X, self.y)

        
        print(self.svm.decision_function(self.x1))
        print(self.svm.predict(self.x1))
        print(self.svm.support_vectors_ )
        print(self.svm.score(self.x1, self.y))
        
        self.exportModel('modelterbaru.pkl', featurelist)

    def cetak(self, pesan):
        print(pesan)
        self.update.emit(pesan)