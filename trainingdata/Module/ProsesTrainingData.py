
import re
import sys
import csv
import math
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style

style.use("ggplot")

from sklearn import svm
from Utility import tfidf

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer


class TrainingData:
    def __init__(self):
       self.runProcessTraining()

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
    
    def runProcessTraining(self):
        positive = 'positif'
        negative = 'negative'
        neutral = 'neutral'

        st = open('data/feature_list/id-stopwords.txt', 'r')
        stopWords = self.getStopWordList('data/feature_list/id-stopwords.txt')

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

        x1 = []
        x2 = []

        label = []

        tfidfDocument = tfidf.TfIdf()

        #Read the tweets one by one and process it
        ##inpTweets = csv.reader(open('data/sampleTweets.csv', 'rb'), delimiter=',', quotechar='|')

        csvfile = open('data/sampleTweets.csv', "r")
        inpTweets = csv.reader(csvfile)
        tweets = []

        # start loop
        for i, row in enumerate(inpTweets):
            sentiment = row[0].replace('|','')
            tweet = row[1].replace('|', '')
            
            print("preprocessing data ke ", i," tweet : ", tweet)

            #tahap preprocessing
            processedTweet = self.processTweet(tweet)
            featureVector = self.preprocessingData(processedTweet, stopWords)
            tweets.append((featureVector, sentiment))

             # tahap binary
            tfidfDocument.add_document(i,featureVector)

            if (sentiment == 'positive' ):
                document['positif'].append(tweet)

                for feature in featureVector:
                    featurelist['positif'].append(feature)

                featurelist['positif'] = list(set(featurelist['positif']))

                label.append(1)

            if (sentiment == 'negative' ):
                document['negative'].append(tweet)

                for feature in featureVector:
                   featurelist['negative'].append(feature)

                featurelist['negative'] = list(set(featurelist['negative']))

                label.append(-1)

            if (sentiment == 'neutral' ):
                document['neutral'].append(tweet)

                for feature in featureVector:
                   featurelist['neutral'].append(feature)

                featurelist['neutral'] = list(set(featurelist['neutral']))

                label.append(0)
        
        
        for i, feature in enumerate(featurelist):
            print("generating tf idf per feature : ", feature)
            tfidfresult[feature] = tfidfDocument.similarities(featurelist[feature])
            
            for x in tfidfresult[feature]:
                tfidfweight[feature].append(x[1])

            print(tfidfresult[feature])
            print(tfidfweight[feature])

            print('\n\n\n')

     
        # merubah ke variable yang bisa diterima oleh svm
        for i, row in enumerate(tfidfweight[positive]):
            a = [tfidfresult[positive][i][1],tfidfresult[negative][i][1], tfidfresult[neutral][i][1]]

            x1.append(a)

        print(x1)

        X = x1
        y = label

        clf = svm.SVC(kernel='linear', C = 1.0, decision_function_shape='ovo')
        clf.fit(X,y)
        
        print(clf.predict([1.2828054298642533, 0.12362637362637363, 0.04013377926421405]))
        print(clf.predict([10.58,10.76,0.04013377926421405]))
        w = clf.coef_[0]
        print(w)

        a = -w[0] / w[1]

        xx = np.linspace(0,12)
        yy = a * xx - clf.intercept_[0] / w[1]

        h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

       # plt.scatter(X[:, 0], X[:, 1], c = y)
       # plt.legend()
       # plt.show()



