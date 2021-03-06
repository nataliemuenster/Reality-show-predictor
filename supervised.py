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
import domain_specific as ds
import word_vector as wv
import util
from sklearn.metrics import precision_recall_fscore_support

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

#run supervised with: python supervised.py <directory name of data> <file name of classifications>
#python supervised.py ../cs221-data/read-data/ ./labeled_data.txt majority/nb/ds/wv
def main(argv):
    if len(argv) < 4:
        print >> sys.stderr, 'Usage: python supervised.py <data directory name> <labels file name> <algorithm>'
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict)
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    
    random.shuffle(labeledData)
    numTrain = 4*len(labeledData) / 5 #training set = 80% of the data
    numCorrect = 0
    numTotal = 0
    testResults = ([],[]) #Y, prediction
    #if argv[3] == 'wv':
    #    klassList = []
    #    newLabeledList = []
    #    with open('./lol.txt', 'r') as file:
    #        counter = 1
    #        for line in file:
    #            #print line
    #            num = int(line.split(":")[0])
    #            for dataPoint in labeledData:
    #                if dataPoint[0] == num:
    #                    newLabeledList.append(dataPoint)
    #                    break
    #            if counter == numTrain:
    #                random.shuffle(newLabeledList)
    #            counter += 1
    #    labeledData = newLabeledList
        #print labeledData
    if argv[3] == "majority":
        #classify every example the same
        trainSet = labeledData[:numTrain]
        testSet = labeledData[numTrain:]
        numLeft = 0
        numRight = 0
        for ex in trainSet:
            if ex[2] == -1:
                numLeft += 1
            else: numRight += 1
        print "NUMLEFT: " + str(numLeft) + ",  NUMRIGHT: " + str(numRight)
        majorityKlass = -1 if numLeft > numRight else 1
        for t in testSet:
            numTotal += 1
            if t[2] == majorityKlass:
                numCorrect += 1


    elif argv[3] == "nb":
        classifier = nb_opt.NaiveBayes()
        #classifier.setN(2)
        for i in xrange(len(labeledData)):
            dataPoint = labeledData[i]
            if i < numTrain: #training set
                classifier.train(dataPoint[2], dataPoint[1]['text'], False) #only uses text of the example
            else: #dev set -- only classify once training data all inputted
            #need dev/val and test sets??
                classification = classifier.classify(dataPoint[1]['text'], False)
                numTotal += 1
                #print classification
                if classification == dataPoint[2]:
                    numCorrect += 1
                testResults[0].append(dataPoint[2])
                testResults[1].append(classification)
        precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
        print "Baseline NB TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)

    elif argv[3] == "ds":
        classifier = ds.LinearClassifier(20) #(numIterations, eta)
        trainSet = labeledData[:numTrain] #training set
        testSet = labeledData[numTrain:]
        weights = classifier.perform_sgd(trainSet) #uses text and title of the example
        #dev set -- only classify once training data all inputted
        for ex in trainSet:
            classification = classifier.classify(ex[1], weights)
            numTotal += 1
            #print classification
            if classification == ex[2]:
                numCorrect += 1
        print "TRAIN: " + str(float(numCorrect) / numTotal)
        numCorrect = 0
        numTotal = 0
        for ex in testSet:
            classification = classifier.classify(ex[1], weights)
            numTotal += 1
            #print classification
            if classification == ex[2]:
                numCorrect += 1
            testResults[0].append(ex[2])
            testResults[1].append(classification)
        precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
        print "Baseline domain specific TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)

    #load pretrained word vector data and store article vectors. Only needd to be run once ever as preprocessing.
    elif argv[3] == "load_wv":
        util.vectorizeArticles(dataList)
        return

    elif argv[3] == "wv":
        classifier = wv.WordVector(len(dataList),150) #(numArticles, numIterations, eta)
        trainSet = labeledData[:numTrain] #training set
        testSet = labeledData[numTrain:] 
        weights = classifier.perform_sgd(trainSet) #uses text and title of the example
        #dev set -- only classify once training data all inputted
        numCorrect = 0
        numTotal = 0
        for ex in trainSet:
            classification = classifier.classify(ex, weights)
            numTotal += 1
            #print classification
            if classification == ex[2]:
                numCorrect += 1
        print "TRAIN: " + str(float(numCorrect) / numTotal)
        #precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
        #print "Baseline domain specific TRAIN scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
        numCorrect = 0
        numTotal = 0
        for ex in testSet:
            classification = classifier.classify(ex, weights)
            numTotal += 1
            #print classification
            if classification == ex[2]:
                numCorrect += 1
            testResults[0].append(ex[2])
            testResults[1].append(classification)
        print float(numCorrect) / numTotal
        precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
        print "Baseline domain specific TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
    else:
        print >> sys.stderr, 'Usage: python readDate.py <directory name> <labels file name> <algorithm> ("nb" or "ds")'

    print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)
    #return float(numCorrect) / numTotal


if __name__ == '__main__':
    #score = 0.0
    for _ in xrange(10):
        main(sys.argv)
    #print score / 10

