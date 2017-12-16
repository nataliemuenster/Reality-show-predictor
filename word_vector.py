import random
import collections
import math
import sys
import os
import csv
import string
import util
import numpy as np
import cPickle as pkl

class WordVector:
    def __init__(self, numArticles, numIters = 20, eta = .02):
        self.stopWords = self.readStopWordsFile('./english.stop')
        self.featureExtractor = self.getArticleFeatures
        self.articleVectorsFileName = "article_word_vectors_wo_stop_binary_words.txt"
        self.articleVectors = self.readArticleVectorsFile()
        self.numIters = numIters
        self.eta = eta
        self.numArticles = numArticles

    def readStopWordsFile(self, fileName):
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        return ('\n'.join(contents)).split()

    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopWords and word.strip() != '':
                filtered.append(word)
        return filtered

    def readArticleVectorsFile(self):
        articleVectors = {}
        articleNum = 0
        f = open(self.articleVectorsFileName, 'r')
        numRead = 0
        for line in f:
            line = line.strip('\n')
            parts = line.split(':')
            if parts[1] == "": 
            	articleVectors[int(parts[0])] = {}
            	continue
            vect = parts[1].split(' ')
            features = {}
            for i in xrange(len(vect)):
                features[i] = float(vect[i])
            articleVectors[int(parts[0])] = features
            articleNum += 1
            numRead += 1
        f.close()
        #print "total num articles: " + str(articleNum)
        return articleVectors


    def getArticleFeatures(self, exampleNum):
        return self.articleVectors[exampleNum]
        

    def perform_sgd(self, trainExamples):
        weights = {}
        for i in xrange(self.numIters):
            for example in trainExamples:
                #phi(x)
                featureVector = self.featureExtractor(example[0]) #pass in example number
                #y
                yValue = example[2]

                margin = 0
                for key in featureVector:
                    if weights.has_key(key):
                        margin += featureVector[key]*weights[key]
                margin *= yValue
                if margin < 1:
                    for key in featureVector:
                        if weights.has_key(key):
                            weights[key] = weights[key] + self.eta * yValue * featureVector[key]
                        else:
                            weights[key] = self.eta * yValue * featureVector[key]
        return weights

    def classify(self, example, weights):
        dotProduct = util.dotProduct(self.featureExtractor(example[0]), weights)
        if dotProduct >= 0:
            return 1
        else:
            return -1


