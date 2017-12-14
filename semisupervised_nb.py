import random
import collections
import math
import sys
import numpy as np
import os
import csv
import string
import naive_bayes as nb
import naive_bayes_optimized as nb_opt
import semisupervised as ss
import util
import time 
import cross_validation as cv
from sklearn.metrics import precision_recall_fscore_support

def train_semisupervised_nb(trainSet, unlabeledData):
    secondNB = nb_opt.NaiveBayes()
    for i, dataPoint in enumerate(trainSet):
        secondNB.train(dataPoint[2], dataPoint[1]['text'], True)
    random.shuffle(unlabeledData)
    unlabeledLiberal = 0
    unlabeledConservative = 0
    for i in range(len(unlabeledData)/10): #only use 1% of the full unlabeled dataset
        dataPoint = unlabeledData[i]
        classification = secondNB.classify(dataPoint[1]['text'], True)
        secondNB.train(classification, dataPoint[1]['text'], True)
        #print i
        #print classification
        if classification == -1:
            unlabeledLiberal += 1
        else:
            unlabeledConservative += 1
    return secondNB


#semisupervised_nb.py
def main(argv):
	#argv[1] = "../cs221-data/read-data/", argv[2] = "./labeled_data.txt"
    if len(argv) < 3:
        print >> sys.stderr, 'Usage: python semisupervised_nb.py <data directory name> <labels file name>' #what is this?
        sys.exit(1)
    cross_validation = False
    if len(argv) > 3 and argv[3] == "cv": cross_validation = True
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    classifier = nb.NaiveBayes()

    print "semi-supervised"
    # random.shuffle(labeledData)
    # numTrain = 4*len(labeledData) / 5 #training set = 80% of the data
    numCorrect = 0
    numTotal = 0
    testResults = ([],[]) #Y, prediction
    secondNB = nb.NaiveBayes()
    
    # for i in range(len(labeledData)):
    #     dataPoint = labeledData[i]
    #     if i < numTrain: #training set
    #         secondNB.train(dataPoint[2], dataPoint[1]['text'])
    #     else:
    #         break
    # random.shuffle(unlabeledData)
    # unlabeledLiberal = 0
    # unlabeledConservative = 0
    # for i in range(len(unlabeledData)/100): #only use 1% of the full unlabeled dataset
    #     dataPoint = unlabeledData[i]
    #     classification = secondNB.classify(dataPoint[1]['text'])
    #     secondNB.train(classification, dataPoint[1]['text'])
    #     #print i
    #     #print classification
    #     if classification == -1:
    #         unlabeledLiberal += 1
    #     else:
    #         unlabeledConservative += 1
    results = []
    folds = cv.getFolds(labeledData)

    for i, fold in enumerate(folds):
        trainSet = cv.getTrainFolds(folds, i)
        testSet = cv.getTestFolds(folds, i)
        numTotal = 0
        numCorrect = 0
        secondNB = train_semisupervised_nb(trainSet, unlabeledData)
        for dataPoint in testSet:
            classification = secondNB.classify(dataPoint[1]['text'], True)
            numTotal += 1
            #print classification
            if classification == dataPoint[2]:
                numCorrect += 1
            testResults[0].append(dataPoint[2])
            testResults[1].append(classification)
        precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
        print "Semisupervised NB scores for fold %d:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (i, precision, recall, fscore)
        print "numCorrect for fold: " + str(i) + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)
        results.append(numCorrect)
        if not cross_validation: break
    average = sum(results) / float(len(results))
    return (results, average, numTotal)

    # trainSet = getTrainFolds()
    # testSet = getTestFolds()
    # secondNB = train_semisupervised_nb(trainSet, unlabeledData)
    # for i, dataPoint in testSet:
    #     classification = secondNB.classify(dataPoint[1]['text'])
    #     numTotal += 1
    #     #print classification
    #     if classification == dataPoint[2]:
    #         numCorrect += 1
    #     testResults[0].append(dataPoint[2])
    #     testResults[1].append(classification)
    # results.append(numCorrect)
    # precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
    # print "semisupervised NB TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
    # print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)
    # return (results, numCorrect, numTotal)

if __name__ == '__main__':
    results = []
    for _ in xrange(2): #change to 10 if running w/o cross-validation
    	testResults, testAverage, numTotal = main(sys.argv)
        results += testResults
    resultPercentages = [(float(result)/float(numTotal)) for result in results]
    beginBar, endBar, average = cv.computeErrorBar(resultPercentages)
    print "Error Bar: (%.5f, %.5f) with average accuracy %.5f" % (beginBar, endBar, average)

