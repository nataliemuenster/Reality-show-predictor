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
import sgd as sgd
import semisupervised as ss
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

def getFolds(dataList, nFolds=5):
    numArticles = len(dataList)
    random.shuffle(dataList)
    numPerFold = numArticles / nFolds
    folds = []
    for i in range(nFolds):
        startIndex = numPerFold * i
        if i == nFolds - 1:
            folds.append(dataList[startIndex:])
        else:
            endIndex = startIndex + numPerFold
            folds.append(dataList[startIndex:endIndex])
    return folds

#run cross_validation with: python cross_validation.py <directory name of data> <file name of classifications>
#python cross_validation.py ../cs221-data/read-data/ ./labeled_data.txt nb/sgd 
def main(argv):
    if len(argv) < 4:
        print >> sys.stderr, 'Usage: python baseline.py <data directory name> <labels file name> <algorithm>'
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    
    random.shuffle(labeledData)
    folds = getFolds(labeledData)
    numTrain = 4*len(labeledData) / 5 #training set = 80% of the data
    numCorrect = 0
    numTotal = 0
    testResults = ([],[]) #Y, prediction

    def getTestFolds(testIndex):
        return folds[testIndex]

    def getTrainFolds(testIndex):
        trainFolds = []
        for i, fold in enumerate(folds):
            if i != testIndex:
                trainFolds += fold
        return trainFolds

    #list of percentage correct in each run
    results = []
    if argv[3] == "majority":
        #crossValidation:
        for i, fold in enumerate(folds):
            trainSet = getTrainFolds(i)
            testSet = getTestFolds(i)
            numLeft = 0
            numRight = 0
            for ex in trainSet:
                if ex[2] == -1: numLeft += 1
                else: numRight += 1
            print "NUMLEFT: " + str(numLeft) + ",  NUMRIGHT: " + str(numRight)
            majorityKlass = -1 if numLeft > numRight else 1
            for t in testSet:
                numTotal += 1
                if t[2] == majorityKlass:
                    numCorrect += 1
            results.append(numCorrect)

    elif argv[3] == "nb":
        classifier = nb_opt.NaiveBayes()
        #cross-validation
        for i, fold in enumerate(folds):
            trainSet = getTrainFolds(i)
            testSet = getTestFolds(i)
            for i, dataPoint in enumerate(trainSet):
                classifier.train(dataPoint[2], dataPoint[1]['text'])
            for i, dataPoint in enumerate(testSet):
                classification = classifier.classify(dataPoint[1]['text'])
                numTotal += 1
                #print classification
                if classification == dataPoint[2]:
                    numCorrect += 1
                testResults[0].append(dataPoint[2])
                testResults[1].append(classification)
            precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
            print "Baseline NB TEST scores for fold %d:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (i, precision, recall, fscore)
            results.append(numCorrect)

    elif argv[3] == "sgd":
        classifier = sgd.SGD(20) #(numIterations, eta)
        for i, fold in enumerate(folds):
            trainSet = getTrainFolds(i) #training set
            testSet = getTestFolds(i) #need dev and test set??
            weights = classifier.perform_sgd(trainSet) #uses text and title of the example
            #dev set -- only classify once training data all inputted
            for ex in testSet:
                classification = classifier.classify(ex[1], weights)
                numTotal += 1
                print classification
                if classification == ex[2]:
                    numCorrect += 1
                testResults[0].append(ex[2])
                testResults[1].append(classification)
            precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
            print "Baseline SGD TEST scores for fold %d:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (i, precision, recall, fscore)
            results.append(numCorrect)

    else:
        print >> sys.stderr, 'Usage: python readDate.py <directory name> <labels file name> <algorithm> ("nb" or "sgd")'

    average = sum(results) / float(len(folds))
    print "Average numCorrect: " + str(average) + " numTotal: " + str(numTotal) + " percentage: " + str(float(average) / float(numTotal))
    return (results, average, numTotal)

    #print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)

def computeErrorBar(resultList):
    average = float(sum(resultList)) / (float(len(resultList)))
    squaredDiffs = 0
    for result in resultList: squaredDiffs += ((result - average) ** 2)
    stdDev = math.sqrt(squaredDiffs/float(len(resultList)))
    return (average - stdDev, average + stdDev, average)

if __name__ == '__main__':
    #numCorrect for each individual fold in each test (will be 50 total)
    results = []
    #average of numCorrect for each test (will be 10 total)
    averages = []
    for _ in xrange(10):
        testResults, testAverage, numTotal = main(sys.argv)
        results += testResults
        averages.append(testAverage)
    # use results or averages?
    resultPercentages = [(float(result)/float(numTotal)) for result in results]
    averagePercentages = [(float(avg)/float(numTotal)) for avg in averages]
    beginBar, endBar, average = computeErrorBar(resultPercentages)
    print "Error Bar: (%.3f, %.3f) with average %.3f" % (beginBar*100, endBar*100, average*100)


