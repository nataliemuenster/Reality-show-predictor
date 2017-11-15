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