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



def createClassDict(classificationsFile):
	classDict = {} #sparseVector
	with open(classificationsFile, 'r') as file:
		for line in file:
			klass = line.split(":")
			classDict[int(klass[0])] = int(klass[1])
	return classDict

#directoryName = "../cs221-data/read-data/"
def readFiles(directoryName, classificationDict = {}):
	dataList = []

	firstLine = True
	#'../cs221-data/read-data/':
	csv.field_size_limit(sys.maxsize)
	exampleNum = 0 #examples are zero-indexed
	labeledExNums = [exNum for exNum in classificationDict] #what happens if None?
	for fileName in os.listdir(directoryName):
		if fileName == '.DS_Store':
			continue
		fileName = os.path.join(directoryName, fileName)
		with open(fileName, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				klass = None
				if len(classificationDict.items()) > 0:
					if exampleNum in classificationDict:
						klass = classificationDict[exampleNum]
					else: klass = None
				else:
					klass = random.choice([-1, 1])
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
						}, klass))
					exampleNum += 1
		csvfile.close()
	return dataList


def separateLabeledExamples(dataList): #could use classificationDict here instead
	labeledExamples = [] #no dict/sparse vector needed
	unlabeledExamples = []
	for ex in dataList:
		if ex[1] == None:
			unlabeledExamples.append(ex)
		else:
			labeledExamples.append(ex) #example, klass
	return labeledExamples, unlabeledExamples


def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())
        
