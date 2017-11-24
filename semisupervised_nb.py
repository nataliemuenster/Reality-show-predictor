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
import util
import time 

#semisupervised_nb.py
def main(argv):
	#argv[1] = "../cs221-data/read-data/", argv[2] = "./labeled_data.txt"
    if len(argv) < 3:
        print >> sys.stderr, 'Usage: python semisupervised_nb.py <data directory name> <labels file name>' #what is this?
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    classifier = nb.NaiveBayes()

    print "semi-supervised"
    random.shuffle(labeledData)
    numTrain = 4*len(labeledData) / 5 #training set = 80% of the data
    numCorrect = 0
    numTotal = 0
    secondNB = nb.NaiveBayes()
    
    for i in range(len(labeledData)):
        dataPoint = labeledData[i]
        if i < numTrain: #training set
            secondNB.train(dataPoint[1], dataPoint[0]['text'])
        else:
            break
    random.shuffle(unlabeledData)
    unlabeledLiberal = 0
    unlabeledConservative = 0
    for i in range(len(unlabeledData)/100): #only use 1% of the full unlabeled dataset
        dataPoint = unlabeledData[i]
        classification = secondNB.classify(dataPoint[0]['text'])
        secondNB.train(classification, dataPoint[0]['text'])
        #print i
        #print classification
        if classification == -1:
            unlabeledLiberal += 1
        else:
            unlabeledConservative += 1


    for i in range(numTrain, len(labeledData)):
        dataPoint = labeledData[i]
        classification = secondNB.classify(dataPoint[0]['text'])
        numTotal += 1
        #print classification
        if classification == dataPoint[1]:
            numCorrect += 1
    print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)



if __name__ == '__main__':
    for _ in xrange(10):
    	main(sys.argv)
