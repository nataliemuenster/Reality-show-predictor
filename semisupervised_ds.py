import random
import collections
import math
import sys
import numpy as np
import os
import csv
import string
import naive_bayes as nb
import domain_specific as ds
import util
import time
import cross_validation as cv
from sklearn.metrics import precision_recall_fscore_support


def train_semisupervised_ds(trainSet, unlabeledData, classifier):
    weights = classifier.perform_sgd(trainSet)
    subsequentTrainSet = list(trainSet)
    random.shuffle(unlabeledData)
    unlabeledLiberal = 0
    unlabeledConservative = 0
    #everyTenFlag = (len(unlabeledData) / 10) % 100
    stopTrain = 10
    for i in range(len(unlabeledData) / 100): # Use 1% of data for now
        if i == 100:
            stopTrain = 50
        if i == 500:
            stopTrain = 100
        if i == 1500:
            stopTrain = 1000
        if i == 4500:
            stopTrain = 2500

        classification = classifier.classify(unlabeledData[i][1], weights)
        if classification == -1:
            unlabeledLiberal += 1
        else:
            unlabeledConservative += 1
        subsequentTrainSet.append((unlabeledData[i][0], unlabeledData[i][1], classification))
        if (i + 1) % stopTrain == 0:
            weights = classifier.perform_sgd(subsequentTrainSet)
    #if not everyTenFlag == 0:
    #    weights = classifier.perform_sgd(subsequentTrainSet)
    return weights



def main(argv):
	#argv[1] = "../cs221-data/read-data/", argv[2] = "./labeled_data.txt"
    if len(argv) < 3:
        print >> sys.stderr, 'Usage: python semisupervised_ds.py <data directory name> <labels file name>' #what is this?
        sys.exit(1)
    cross_validation = False
    if len(argv) > 3 and argv[3] == "cv": cross_validation = True
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    #classifier = ds.LinearClassifier(10)

    print "semi-supervised"
    # random.shuffle(labeledData)
    # numTrain = 4*len(labeledData) / 5 #training set = 80% of the data
    numCorrect = 0
    numTotal = 0
    testResults = ([],[]) #Y, prediction
    folds = cv.getFolds(labeledData)
    # trainSet = labeledData[:numTrain] #training set
    # testSet = labeledData[numTrain:] #need dev and test set??
    # weights = classifier.perform_sgd(trainSet)

    # subsequentTrainSet = list(trainSet)
    # random.shuffle(unlabeledData)
    # unlabeledLiberal = 0
    # unlabeledConservative = 0
    # everyTenFlag = (len(unlabeledData) / 1000) % 50
    # for i in range(len(unlabeledData) / 1000): # Use 1% of data for now
    #     #print "Ex being passed in unlabeled classify is: " + str(unlabeledData[i][1])
    #     classification = classifier.classify(unlabeledData[i][1], weights)
    #     if classification == -1:
    #         unlabeledLiberal += 1
    #     else:
    #         unlabeledConservative += 1
    #     subsequentTrainSet.append((unlabeledData[i][0], unlabeledData[i][1], classification))
    #     if (i + 1) % 50 == 0:
    #         weights = classifier.perform_sgd(subsequentTrainSet)
    # if not everyTenFlag == 0:
    #     weights = classifier.perform_sgd(subsequentTrainSet)
    for i, fold in enumerate(folds):
        trainSet = cv.getTrainFolds(folds, i)
        testSet = cv.getTestFolds(folds, i)
        numTotal = 0
        numCorrect = 0
        classifier = ds.LinearClassifier(10)
        weights = train_semisupervised_ds(trainSet, unlabeledData, classifier)
        for ex in testSet:
            #print "Ex being passed in labeled classify is: " + str(ex[1])
            classification = classifier.classify(ex[1], weights)
            numTotal += 1
            print classification
            if classification == ex[2]:
                numCorrect += 1
            testResults[0].append(ex[2])
            testResults[1].append(classification)

        precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
        print "semisupervised domain specific TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
        print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)
        results.append(numCorrect)
        if not cross_validation: break
    average = sum(results) / float(len(results))
    return (results, average, numTotal)



if __name__ == '__main__':
    results = []
    for _ in xrange(2):
        testResults, testAverage, numTotal = main(sys.argv)
        results += testResults
    resultPercentages = [(float(result)/float(numTotal)) for result in results]
    beginBar, endBar, average = cv.computeErrorBar(resultPercentages)
    print "Error Bar: (%.5f, %.5f) with average accuracy %.5f" % (beginBar, endBar, average)
