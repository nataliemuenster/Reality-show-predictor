from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import random
import collections
import math
import sys
import numpy as np
import os
import csv
import random
import string
import naive_bayes as nb

stopList = []
for line in open('./english.stop', 'r'):
	stopList.append(line)
stopList = set(stopList)


def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def get_unigrams(text):
	text = text.translate(None, string.punctuation).lower()
	unigrams = collections.defaultdict(lambda:[0, 0])
	textList = text.split()
	for word in textList:
		unigrams[word][0] += 1
	for word in unigrams:
		unigrams[word][1] = math.log(unigrams[word][0]) - math.log(len(textList))
	return unigrams


def readFiles(directoryName):
	print directoryName
	dataList = []
	if directoryName == '../cs221-data/congressional-votes-small/':
		for fileName in os.listdir(directoryName):
			if fileName == '.DS_Store':
				continue
			fileName = os.path.join(directoryName, fileName)
			dataFile = open(fileName, 'r')
			textFile = dataFile.read().replace('\n', '')
			dataList.append(({'text': textFile, 'type': 'speech'}, random.randint(-1, 1)))

	firstLine = True

	if directoryName == '../cs221-data/fake-data/':
		csv.field_size_limit(sys.maxsize)
		for fileName in os.listdir(directoryName):
			if fileName == '.DS_Store':
				continue
			print fileName
			fileName = os.path.join(directoryName, fileName)
			print fileName
			with open(fileName, 'rb') as csvfile:
				reader = csv.reader(csvfile)
				for line in reader:
					if firstLine:
						firstLine = False
					else:
						fullText = line[4] + ' ' + line[5].replace('\n', '')
						dataList.append(({
								'author': line[2],
								'published': line[3],
								'title': line[4],
								'text': line[5].replace('\n', ''),
								'site_url': line[8],
								'type': line[19]
							}, random.randint(-1, 1)))
						
	firstLine = True
	if directoryName == '../cs221-data/read-data/':
		csv.field_size_limit(sys.maxsize)
		for fileName in os.listdir(directoryName):
			if fileName == '.DS_Store':
				continue
			fileName = os.path.join(directoryName, fileName)
			with open(fileName, 'rb') as csvfile:
				reader = csv.reader(csvfile)
				for line in reader:
					if firstLine:
						firstLine = False
					else:
						fullText = line[2] + ' ' + line[9].replace('\n', '')
						dataList.append(({
								'title': line[2],
								'publication': line[3],
								'author': line[4],
								'date': line[5],
								'site_url': line[8],
								'text': line[9].replace('\n', '')
							}, random.randint(-1, 1)))
	return dataList


#trainExamples = util.readExamples('???')
#valExamples = util.readExamples('???')
#testExamples = util.readExamples('???')

def main(argv):
    if len(argv) < 2:
        print >> sys.stderr, 'Usage: python readDate.py <directory name>'
        sys.exit(1)
    dataList = readFiles(argv[1])
    classifier = nb.NaiveBayes()
    random.shuffle(dataList)
    numTrain = 4 * len(dataList) / 5 
    numCorrect = 0
    numTotal = 0
    #print len(dataList)
    #print numTrain
    for i in range(len(dataList)):
    	dataPoint = dataList[i]
        if i < numTrain:
        	classifier.train(dataPoint[1], dataPoint[0]['text'])
        else:
        	classification = classifier.classify(dataPoint[0]['text'])
        	numTotal += 1
        	#print classification
        	if classification == dataPoint[1]:
        		numCorrect += 1
    print "numCorrect: " + str(numCorrect) + ' numTotal: ' + str(numTotal) + ' percentage: ' + str(float(numCorrect) / numTotal)

if __name__ == '__main__':
    main(sys.argv)

