import random
import collections
import math
import sys
import numpy as np
import os
import csv
import string
import naive_bayes as nb
import semisupervised as ss
import sgd as sgd
import util
import time
from sklearn.metrics import precision_recall_fscore_support

#semisupervised_nb.py
def main(argv):
	#argv[1] = "../cs221-data/read-data/", argv[2] = "./labeled_data.txt"
    if len(argv) < 3:
        print >> sys.stderr, 'Usage: python semisupervised_sgd.py <data directory name> <labels file name>' #what is this?
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    classifier = sgd.SGD(10)

    print "semi-supervised"
    random.shuffle(labeledData)
    numTrain = 4*len(labeledData) / 5 #training set = 80% of the data
    numCorrect = 0
    numTotal = 0
    testResults = ([],[]) #Y, prediction
    trainSet = labeledData[:numTrain] #training set
    testSet = labeledData[numTrain:] #need dev and test set??
    weights = classifier.perform_sgd(trainSet)

    subsequentTrainSet = list(trainSet)
    random.shuffle(unlabeledData)
    unlabeledLiberal = 0
    unlabeledConservative = 0
    everyTenFlag = (len(unlabeledData) / 1000) % 50
    for i in range(len(unlabeledData) / 1000): # Use 1% of data for now
        #print "Ex being passed in unlabeled classify is: " + str(unlabeledData[i][1])
        classification = classifier.classify(unlabeledData[i][1], weights)
        if classification == -1:
            unlabeledLiberal += 1
        else:
            unlabeledConservative += 1
        subsequentTrainSet.append((unlabeledData[i][0], unlabeledData[i][1], classification))
        if (i + 1) % 50 == 0:
            weights = classifier.perform_sgd(subsequentTrainSet)
    if not everyTenFlag == 0:
        weights = classifier.perform_sgd(subsequentTrainSet)


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
    print "semisupervised SGD TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
    print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)



if __name__ == '__main__':
    for _ in xrange(10):
    	main(sys.argv)
