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
    def __init__(self, numIters = 20, eta = 0.01):
        #self.wordVectDict = self.readVectorsFile('../cs221-data/glove.42B.300d.txt')
        self.stopWords = self.readStopWordsFile('./english.stop')
        self.featureExtractor = getArticleVector
        self.articleVectorsFileName = "article_word_vectors.txt"
        self.articleVectors = self.readArticleVectorsFile()

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
    	articleVectors = []
    	articleNum = 0
        with open(self.articleVectorsFileName, 'rb') as f:
    		while True:
		        try:
		            vect = pkl.load(f)
		            print vect
		            articleVectors.append(vect)
		            articleNum += 1
		        except EOFError:
		            break
        f.close()
        return articleVectors


    def getArticleVector(self, example):
    	return self.articleVectors[example[0]]

    def perform_sgd(self, trainExamples):
        weights = []
        #print trainExamples[0]
        for i in range(self.numIters):
            print "iteration " + str(i)
            for example in trainExamples:
                #phi(x)
                #print "Ex feature vectorized is: " + str(example[1])
                featureVector = self.featureExtractor(example)
                #y
                yValue = example[2]
                #calculates phi(x)y
                for j in xrange(len(featureVector)):
                    featureVector[j] *= yValue

                gradientLoss = []
                #if w dot phi(x)y < 1, we use -phi(x)y (which cancels to just phi(x)y)
                #if util.dotProduct(weights, featureVector) < 1:
                dotProduct = sum([weights[i]*featureVector[i] for i in xrange(len(featureVector))])
                if dotProduct < 1:
                    gradientLoss = featureVector
                #util.increment(weights, self.eta, gradientLoss)
                weights = [weights[f] + v * self.eta for f, v in d2.items()]
        
        return weights


    def classify(self, example, weights):
        featureVector = self.featureExtractor(example)
        dotProduct = sum([weights[i]*featureVector[i] for i in xrange(len(featureVector))])
        if dotProduct >= 0:
            return 1
        else:
            return -1


