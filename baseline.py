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
import word_vector as wv
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

#run baseline with: python baseline.py <directory name of data> <file name of classifications>
#python baseline.py ../cs221-data/read-data/ ./labeled_data.txt majority/nb/sgd/wv
def main(argv):
    if len(argv) < 4:
        print >> sys.stderr, 'Usage: python baseline.py <data directory name> <labels file name> <algorithm>'
        sys.exit(1)
    classificationDict = util.createClassDict(argv[2])
    dataList = util.readFiles(argv[1], classificationDict) #if no classificationDict passed in, randomized
    labeledData, unlabeledData = util.separateLabeledExamples(dataList) 
    
    random.shuffle(labeledData)
    numTrain = 4*len(labeledData) / 5 #training set = 80% of the data
    numCorrect = 0
    numTotal = 0
    testResults = ([],[]) #Y, prediction

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
        for i in xrange(len(labeledData)):
            dataPoint = labeledData[i]
            if i < numTrain: #training set
                classifier.train(dataPoint[2], dataPoint[1]['text']) #only uses text of the example
            else: #dev set -- only classify once training data all inputted
            #need dev/val and test sets??
                classification = classifier.classify(dataPoint[1]['text'])
                numTotal += 1
                #print classification
                if classification == dataPoint[2]:
                    numCorrect += 1
                testResults[0].append(dataPoint[2])
                testResults[1].append(classification)
        precision,recall,fscore,support = precision_recall_fscore_support(testResults[0], testResults[1], average='binary')
        print "Baseline NB TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)

    elif argv[3] == "sgd":
        classifier = sgd.SGD(20) #(numIterations, eta)
        trainSet = labeledData[:numTrain] #training set
        testSet = labeledData[numTrain:] #need dev and test set??
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
        print "Baseline SGD TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)

    elif argv[3] == "load_wv":
        wordVectDict = util.writePretrainedWordVectorsToFile()
        print "done pickling pretrained word vector file."
        util.vectorizeArticles(dataList, wordVectDict)
        print "done pickling article word vector file."
        return

    elif argv[3] == "wv":
        classifier = wv.WordVector(20) #(numIterations, eta)
        trainSet = labeledData[:numTrain] #training set
        testSet = labeledData[numTrain:] #need dev and test set??
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
        print "Baseline SGD TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)

    else:
        print >> sys.stderr, 'Usage: python readDate.py <directory name> <labels file name> <algorithm> ("nb" or "sgd")'

    print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)



if __name__ == '__main__':
    for _ in xrange(10):
        main(sys.argv)

