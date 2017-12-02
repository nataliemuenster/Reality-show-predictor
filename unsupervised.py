import random
import collections
import math
import sys
import numpy as np
import os
import csv
import string
import util
import time 
import kmeans
from sklearn.metrics import precision_recall_fscore_support

stopWords = readStopWordsFile('./english.stop')

def readStopWordsFile(fileName):
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    return ('\n'.join(contents)).split()

def filterStopWords(words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in stopWords and word.strip() != '':
        filtered.append(word)
    return filtered

def createWordCountDict(text):
    wordDict = {}
    words = text.translate(None, string.punctuation).lower().split()
    words = filterStopWords(words)
    for word in words:
        if word in wordDict:
            wordDict[word] += 1
        else:
            wordDict[word] = 1
    return wordDict

def findBestSplit(hingeDict, sourceList, startIndex, dataBySource, dataList, liberalSplit, conservativeSplit, wordCountList):
    if startIndex == len(sourceList):

        # calculate hingeLoss by testing on numDev
        # update hingeDict
    else:
        newLiberal = list(liberalSplit)
        newLiberal.append(sourceList[startIndex])
        newConservative = list(conservativeSplit)
        findBestSplit(hingeDict, sourceList, startIndex + 1, dataBySource, dataList, newLiberal, newConservative, wordCountList)

        newLiberal = list(liberalSplit)
        newConservative = list(conservativeSplit)
        newConservative.append(sourceList[startIndex])
        findBestSplit(hingeDict, sourceList, startIndex + 1, dataBySource, dataList, newLiberal, newConservative, wordCountList)


#python unsupervised.py ../cs221-data/read-data/ ./labeled_data.txt 
# Currently Naive Bayes specific, could be generalized
def main(argv):
    if len(argv) < 3:
        print >> sys.stderr, 'Usage: python unsupervised.py <data directory name> <labels file name>'
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized

    # index in list = exampleNum
    # value at index = wordcountDict
    wordCountList = []
    for value in dataList:
        wordCountList.append(createWordCountDict(value[1]['text']))

    labeledData, unlabeledData = util.separateLabeledExamples(dataList)


    random.shuffle(labeledData)

    # Key = Source
    # Value = List of data points
    sourceList = []
    dataBySource = {}
    for dataPoint in dataList:
        if dataPoint['publication'] in dataBySource:
            dataBySource[dataPoint['publication']].append(dataPoint)
        else:
            sourceList.append(dataPoint['publication'])
            dataBySource[dataPoint['publication']] = [dataPoint]
    # Key = hinge loss. The smallest key will be the one with the lowest global loss and best assignment
    # Value = Tuple ([], [], NaiveBayes()) where the first element is the liberal sources, and the second element is the conservative sources
    hingeDict = {}

    # "Dev" Phase
    findBestSplit(hingeDict, sourceList, 0, dataBySource, dataList, [], [], wordCountList)

    # Find best result from "Dev" Phase
    minLoss = float('inf')
    splitTuple = None
    for lossKey in hingeDict:
        if lossKey < minLoss:
            minLoss = lossKey
            splitTuple = hingeDict[lossKey]

    # Test Phase
    numTestCorrect = 0
    total = len(labeledData)
    for i in range(labeledData):
        dataPoint = labeledData[i]
        klass = dataPoint['klass']
        guess = None
        if dataPoint['publication'] in splitTuple[0]:
            guess = -1
        else:
            guess = 1
        if klass == guess:
            numTestCorrect += 1
    print "numCorrect: " + str(numTestCorrect) + ' numTotal: ' + str(total) + ' percentage: ' + str(float(numTestCorrect) / total)














    

if __name__ == '__main__':
    for _ in xrange(10):
        main(sys.argv)