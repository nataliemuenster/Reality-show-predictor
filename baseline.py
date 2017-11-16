
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import random
import collections
import math
import sys
import numpy as np
import os
import csv
import string
import naive_bayes as nb
import sgd as sgd
import semisupervised as ss
import util

stopList = []
for line in open('./english.stop', 'r'):
	stopList.append(line)
stopList = set(stopList)


#aren't using this right now
def get_unigrams(text):
	text = text.translate(None, string.punctuation).lower()
	unigrams = collections.defaultdict(lambda:[0, 0])
	textList = text.split()
	for word in textList:
		unigrams[word][0] += 1
	for word in unigrams:
		unigrams[word][1] = math.log(unigrams[word][0]) - math.log(len(textList))
	return unigrams

#run baseline with: python baseline.py <directory name of data> <file name of classifications>
def main(argv):
	#argv[1] = "../cs221-data/read-data/", argv[2] = "./labeled_data.txt", "nb" or "sgd" 
    if len(argv) < 2:
        print >> sys.stderr, 'Usage: python readDate.py <data directory name> <labels file name> <algorithm>'
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    
    random.shuffle(labeledData)
    numTrain = 4*len(labeledData) / 5 #training set = 80% of the data
    numCorrect = 0
    numTotal = 0
    #print "TOTAL_LABELED: " + str(len(labeledData)) + "TOTAL_UNLABELED: " + str(len(unlabeledData)) + "  NUMTRAIN: " + str(numTrain)
    
    if argv[3] == "nb":
        classifier = nb.NaiveBayes()
        for i in xrange(len(labeledData)):
            dataPoint = labeledData[i]
            if i < numTrain: #training set
                classifier.train(dataPoint[1], dataPoint[0]['text']) #only uses text of the example
            else: #dev set -- only classify once training data all inputted
            #need dev/val and test sets??
                classification = classifier.classify(dataPoint[0]['text'])
                numTotal += 1
                #print classification
                if classification == dataPoint[1]:
                    numCorrect += 1
    elif argv[3] == "sgd":
        classifier = sgd.SGD(2) #CURRENTLY NUMITER = 2 INSTEAD OF 20(?)
        trainSet = labeledData[:numTrain] #training set
        devSet = labeledData[numTrain:] #need dev and test set??
        weights = classifier.perform_sgd(trainSet) #uses text and title of the example
        #dev set -- only classify once training data all inputted
        for ex in devSet:
            classification = classifier.classify(ex[0], weights)
            numTotal += 1
            if classification == ex[1]:
                numCorrect += 1

    else:
        print >> sys.stderr, 'Usage: python readDate.py <directory name> <labels file name> <algorithm> ("nb" or "sgd")'

    print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)



if __name__ == '__main__':
    for _ in xrange(10):
        main(sys.argv)

