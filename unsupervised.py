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

#python unsupervised.py ../cs221-data/read-data/ ./labeled_data.txt 
def main(argv):
    if len(argv) < 3:
        print >> sys.stderr, 'Usage: python unsupervised_kmeans.py <data directory name> <labels file name>'
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    
    random.shuffle(labeledData)
    numDev = len(labeledData) / 2 #training set = 50% of the labeled data
    numTest = len(labeledData) / 2 #training set = 50% of the labeled data
    numDevCorrect = 0
    numTestCorrect = 0
    
    classifier = kmeans.Kmeans(2, 5) #numClusters, maxIterations
    devSet = labeledData[:numDev]
    testSet = labeledData[numDev:]
    
    for ex in dataList: #run the model on all the data, then later check the values for labeled examples
        classifier.createExampleVector(ex)
    
    centroids, clusterAssignments, reconstrLoss = classifier.runKmeans()

    devResults = ([],[]) #Y, prediction
    testResults = ([],[]) #Y, prediction

    print "centroids start at (0,1)"
    for ex in devSet: #determine which cluster corresponds with which classification by randomly assigning and choosing the best
        classification = -1 if clusterAssignments[ex[0]] == centroids[0] else 1
        #print classification
        if classification == ex[2]:
            numDevCorrect += 1
        devResults[0].append(ex[2])
        devResults[1].append(classification)
    print "DEV SET NUM CORRECT: " + str(float(numDevCorrect) / numDev)
    if numDevCorrect > (numDev / 2) else (1,0) #determine whether to switch assignments
    	centroidAssignment = (0,1) 
    	devResults[1] = [1-i for i in devResults]

    precision,recall,fscore,support = precision_recall_fscore_support(devResults[0], devResults[1], average='binary')
	print "Kmeans DEV scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
    print "centroids assigned: " + str(centroidAssignment)
    #evaluate
    for ex in testSet:
        classification = -1 if clusterAssignments[ex[0]] == centroidAssignment[0] else 1
        print classification
        if classification == ex[2]:
            numTestCorrect += 1
        testResults[0].append(ex[2])
        testResults[1].append(classification)

    print "numCorrect: " + str(numTestCorrect) + ' numTotal: ' + str(numTest) + ' percentage: ' + str(float(numTestCorrect) / numTest)
    precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
    print "Kmeans TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)


if __name__ == '__main__':
    for _ in xrange(10):
        main(sys.argv)