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

stopList = set(readFile('./english.stop'))


def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def readFiles(directoryName):
	dataList = []
	if directoryName == '../cs221-data/congressional-votes-small'
		for fileName in os.listdir(directoryName):
			fileName = os.path.join(directoryName, fileName)
			dataFile = open(fileName, 'r')
			dataList.append(({'text': dataFile, 'type': 'speech'}), random.randint(-1, 1))


	firstLine = True
	if directoryName == '../cs221-data/fake-data'
		for fileName in os.listdir(directoryName):
			fileName = os.path.join(directoryName, fileName)
			with open(fileName, 'rb') as csvfile:
				reader = csv.reader(csvfile)
				for line in reader:
					if firstLine:
						firstLine = False
					else:
						dataList.append(({
								'author': line[2],
								'published': line[3],
								'title': line[4],
								'text': line[5],
								'site_url': line[7]
								'type': line[18]
							}, random.randint(-1, 1)))
						
	firstLine = True
	if directoryName == '../cs221-data/read-data'
		for fileName in os.listdir(directoryName):
			fileName = os.path.join(directoryName, fileName)
			with open(fileName, 'rb') as csvfile:
				reader = csv.reader(csvfile)
				for line in reader:
					if firstLine:
						firstLine = False
					else:
						dataList.append(({
								'title': line[2],
								'publication': line[3],
								'author': line[4],
								'date': line[5],
								'site_url': line[8]
								'text': line[9]
							}, random.randint(-1, 1)))


#trainExamples = util.readExamples('???')
#valExamples = util.readExamples('???')
#testExamples = util.readExamples('???')

def main(argv):
    if len(argv) < 2:
        print >> sys.stderr, 'Usage: python readDate.py <directory name>'
        sys.exit(1)
        #'../cs221\ project/congressional\ vote-small/
    dataDict = readFiles(argv[1])

if __name__ == '__main__':
    main(sys.argv)

