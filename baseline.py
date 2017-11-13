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
import util

stopList = []
for line in open('./english.stop', 'r'):
	stopList.append(line)
stopList = set(stopList)

#aren't using this right now
def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

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
	#argv[1] = "../cs221-data/read-data/", argv[2] = "./classifications.txt"
    if len(argv) < 2:
        print >> sys.stderr, 'Usage: python readDate.py <directory name>' #what is this?
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    print "Classifications dict created..."
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    print "Files read in..."
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    classifier = nb.NaiveBayes()
    print "Starting training..."
    random.shuffle(labeledData)
    numTrain = 4 * len(labeledData) / 5 #training set = 80% of the data
    numCorrect = 0
    numTotal = 0
    print "TOTAL_LABELED: " + str(len(labeledData)) + "TOTAL_UNLABELED: " + str(len(unlabeledData)) + "  NUMTRAIN: " + str(numTrain)
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
    print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)

if __name__ == '__main__':
    main(sys.argv)

