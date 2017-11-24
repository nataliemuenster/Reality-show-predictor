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

#python unsupervised.py ../cs221-data/read-data/ ./labeled_data.txt 
def main(argv):
    if len(argv) < 3:
        print >> sys.stderr, 'Usage: python unsupervised_kmeans.py <data directory name> <labels file name>'
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    print labeledData[0]
    return
    
    random.shuffle(labeledData)
    numDev = len(labeledData) / 2 #training set = 50% of the labeled data
    numTest = len(labeledData) / 2 #training set = 50% of the labeled data
    numDevCorrect = 0
    numTestCorrect = 0
    
    classifier = kmeans.Kmeans(2, 5) #numClusters, maxIterations
    devSet = labeledData[:numDev]
    testSet = labeledData[numDev:]
    
    classifier.createExampleVector({"text": "Hello my friend shipwreck"})
    classifier.createExampleVector({"text": "Where did my planet go moon?"})
    classifier.createExampleVector({"text": "Moo"})
    #for ex in dataList:
    #    classifier.createExampleVector(ex[0])
    print "All examples processed."
    centroids, clusterAssignments, reconstrLoss = classifier.runKmeans()
    print "CENTROID1: len and contents" + str(len(centroids[0])) + str(centroids[0])
    print "CENTROID2: len and contents" + str(len(centroids[1])) + str(centroids[1])
    
    for ex in devSet: #determine which cluster corresponds with which classification by randomly assigning and choosing the best
        #print ex
        classification = -1 if clusterAssignments[ex[0]] == centroids[0] else 1 #NEED NUMBER OF EX, NOT EX ITSELF!!
        #print classification
        if classification == ex[1]:
            numDevCorrect += 1
    print "DEV SET NUM CORRECT: " + str(numDevCorrect)
    centroidAssignment = (0,1) if numDevCorrect > numDev / 2 else (1,0) #determine whether to switch assignments

    #evaluate
    for ex in testSet:
        classification = -1 if assignments[ex[0]] == centroidAssignment[0] else 1
        print classification
        if classification == ex[1]:
            numTestCorrect += 1

    print "numCorrect: " + str(numTestCorrect) + ' numTotal: ' + str(numTest) + ' percentage: ' + str(float(numTestCorrect) / numTest)
    


if __name__ == '__main__':
    for _ in xrange(10):
        main(sys.argv)