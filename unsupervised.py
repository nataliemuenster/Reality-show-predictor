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
import naive_bayes_optimized as nb_opt



def readStopWordsFile(fileName):
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    return ('\n'.join(contents)).split()
stopWords = readStopWordsFile('./english.stop')

def filterStopWords(words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in stopWords and word.strip() != '':
        filtered.append(word)
    return filtered

def createWordCountDict(text):
    uniqueWords = set()
    words = text.translate(None, string.punctuation).lower().split()
    words = filterStopWords(words)
    for word in words:
        uniqueWords.add(word)
    return uniqueWords

def findBestSplit(hingeDict, sourceList, startIndex, dataBySource, liberalSplit, conservativeSplit, wordCountList, devData, trainLen):
    if startIndex == len(sourceList):
        #if len(liberalSplit) > 1:# or len(conservativeSplit) < 5:
        #    return
        # Ellen Uncomments this
        #if len(liberalSplit) == 5 or len(conservativeSplit) == 5:
           

        #Jeff uncomments this
        if len(liberalSplit) == 6 or len(conservativeSplit) == 6:

        # natalie uncomments this
        #if len(liberalSplit) == 4 or len(conservativeSplit) == 4:
        # TO DO
        # calculate hingeLoss by testing on numDev
        # update hingeDict
            classifier = nb_opt.NaiveBayes()
            print "foundSplit"
            print liberalSplit
            print conservativeSplit

            #Training Phase
            print "start weights"
            for liberalPoint in liberalSplit:
                data = dataBySource[liberalPoint]
                for dataPoint in data:
                    exampleNum = dataPoint[0]
                    if exampleNum >= trainLen:
                        continue
                    wordCount = wordCountList[exampleNum]
                    classifier.train(-1, wordCount, True)
            print "cons"
            for conservativePoint in conservativeSplit:
                data = dataBySource[conservativePoint]
                for dataPoint in data:
                    exampleNum = dataPoint[0]
                    if exampleNum >= trainLen:
                        continue
                    wordCount = wordCountList[exampleNum]
                    classifier.train(1, wordCount, True)
            # Dev Phase 
            numTestCorrect = 0
            total = len(devData)
            for i in range(total):
                dataPoint = devData[i]
                klass = dataPoint[2]
                exampleNum = dataPoint[0]
                guess = None
                if exampleNum < len(wordCountList):
                    guess = classifier.classify(wordCountList[exampleNum], True)
                    #print "guess " + str(guess)
                    #print "klass " + str(klass)

                else:
                    print "skip"
                    continue
                if guess == klass:
                    numTestCorrect += 1
            accuracy = float(numTestCorrect) / total
            print "finished"
            print accuracy
            hingeDict[accuracy] = (liberalSplit, conservativeSplit, classifier)
    else:
        newLiberal = list(liberalSplit)
        newLiberal.append(sourceList[startIndex])
        newConservative = list(conservativeSplit)
        findBestSplit(hingeDict, sourceList, startIndex + 1, dataBySource, newLiberal, newConservative, wordCountList, devData, trainLen)

        newLiberal = list(liberalSplit)
        newConservative = list(conservativeSplit)
        newConservative.append(sourceList[startIndex])
        findBestSplit(hingeDict, sourceList, startIndex + 1, dataBySource, newLiberal, newConservative, wordCountList, devData, trainLen)


#python unsupervised.py ../cs221-data/read-data/ ./labeled_data.txt 
# Currently Naive Bayes specific, could be generalized
def main(argv):
    if len(argv) < 3:
        print >> sys.stderr, 'Usage: python unsupervised.py <data directory name> <labels file name>'
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized

    wordCountList = []
    counter = 0
    for value in dataList:
        wordCountList.append(createWordCountDict(value[1]['text']))
        counter += 1
        #if counter == 10000:
        #    break
        if counter % 5000 == 0:
            print counter
    print len(wordCountList)

    # index in list = exampleNum
    # value at index = wordcountDict
    print "hi"
    labeledData, unlabeledData = util.separateLabeledExamples(dataList)
    

    #print wordCountList[0]

    devLen = len(labeledData) / 2
    trainLen = len(unlabeledData) / 2
    devData = []
    testData = []
    random.seed(217)
    random.shuffle(labeledData)
    random.shuffle(unlabeledData)
    #print labeledData[0][1]['publication']
    #print unlabeledData[0][1]['publication']
    for i in range(len(labeledData)):
        if i < devLen:
            devData.append(labeledData[i])
        else:
            testData.append(labeledData[i])
    #print devData
    #print testData
    
    print "me"
    # Key = Source
    # Value = List of data points
    sourceList = []
    dataBySource = {}
    for dataPoint in unlabeledData:
        if dataPoint[1]['publication'] in dataBySource:
            dataBySource[dataPoint[1]['publication']].append(dataPoint)
        else:
            print dataPoint[1]['publication']
            sourceList.append(dataPoint[1]['publication'])
            dataBySource[dataPoint[1]['publication']] = [dataPoint]
    # Key = hinge loss. The smallest key will be the one with the lowest global loss and best assignment
    # Value = Tuple ([], [], NaiveBayes()) where the first element is the liberal sources, and the second element is the conservative sources
    hingeDict = {}

    testSourceList = []
    for i in range(len(sourceList)):
        testSourceList.append(sourceList[i])
    print testSourceList
    print sourceList
    # "Dev" Phase
    findBestSplit(hingeDict, testSourceList, 0, dataBySource, [], [], wordCountList, devData, trainLen)
    print "finished"
    print hingeDict
    # Find best result from "Dev" Phase
    bestKey = float('-inf')
    splitTuple = None
    for accuracyKey in hingeDict:
        if accuracyKey > bestKey:
            bestKey = accuracyKey
            splitTuple = hingeDict[accuracyKey]
    print splitTuple
    print bestKey
    # Test Phase
    numTestCorrect = 0
    total = len(testData)
    for i in range(total):
        dataPoint = testData[i]
        klass = dataPoint[2]
        exampleNum = dataPoint[0]
        #print exampleNum
        if exampleNum < len(wordCountList):
            guess = splitTuple[2].classify(wordCountList[dataPoint[0]], True)
            #print guess
        else:
            print "skip"
            continue
        if guess == klass:
            numTestCorrect += 1
    print "TEST DATA   numCorrect: " + str(numTestCorrect) + ' numTotal: ' + str(total) + ' percentage: ' + str(float(numTestCorrect) / total)

if __name__ == '__main__':
    # To speed up, this loop could be pushed inward, so some calculations could be not
    main(sys.argv)