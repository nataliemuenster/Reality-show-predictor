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
import cPickle as pkl


#creates a dictionary mapping all labeled articles to the labels indicated in the specified file
def createClassDict(classificationsFile):
	classDict = {} #sparseVector
	with open(classificationsFile, 'r') as file:
		for line in file:
			klass = line.split(":")
			classDict[int(klass[0])] = int(klass[1])
	return classDict

#directoryName = "../cs221-data/read-data/"
#reads and pre-processes all the articles
def readFiles(directoryName, classificationDict = {}):
	dataList = []

	firstLine = True
	csv.field_size_limit(sys.maxsize)
	exampleNum = 0 #examples are zero-indexed
	labeledExNums = [exNum for exNum in classificationDict]
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
					#balances data set
					if line[3] == 'publication' or line[3] == 'Vox' or line[3] == 'Washington Post' or line[3] == 'Reuters':
						continue
					fullText = line[2] + ' ' + line[9].replace('\n', '')
					dataList.append((exampleNum, {
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


def separateLabeledExamples(dataList):
	labeledExamples = [] #no dict/sparse vector needed
	unlabeledExamples = []
	for ex in dataList:
		if ex[2] == None:
			unlabeledExamples.append(ex)
		else:
			labeledExamples.append(ex) #example, klass
	return labeledExamples, unlabeledExamples

def readStopWordsFile():
    fileName = './english.stop'
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    return ('\n'.join(contents)).split()

#Filters stop words
def filterStopWords(stopWords, words):
    filtered = []
    for word in words:
      if not word in stopWords and word.strip() != '':
        filtered.append(word)
    return filtered

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

def addVectors(a,b):
    if len(a) < len(b):
        c = [float(a[i]) + float(b[i]) for i in xrange(len(a))]
        for j in xrange(len(a), len(b)):
	        c.append(float(b[j]))
        return c
    else:
        c = [float(a[i]) + float(b[i]) for i in xrange(len(b))]
        for j in xrange(len(b), len(a)):
	        c.append(float(a[j]))
        return c

#Builds dictionary of pre-trained word vectors. Accesses this dict to get word vectors of the words 
#for each article and normalize to get the article's feature vector.
def vectorizeArticles(examples):
    wordVectDict = {}
    fr = open("../cs221-data/glove.6B.50d.txt", 'rb') 
    for line in fr:
            vectTerms = line.split()
            wordVectDict[vectTerms[0]] = vectTerms[1:]
    fr.close()
    stopWords = readStopWordsFile()
    print "about to build article vectors"
    fw = open("article_word_vectors_wo_stop_binary_words2.txt",'w')
    num = 0
    
    for ex in examples:
        num += 1
        if num % 500 == 0: print str(num) + " examples done!"
        totalVect = []
        words = ex[1]['text'].translate(None, string.punctuation).lower().split()
        words = filterStopWords(stopWords, words)
        
        words = set(words)
        for word in words:
            if word in wordVectDict:
                newVect = wordVectDict[word]
                totalVect = addVectors(newVect, totalVect)
        normSum = sum(totalVect)
        normed = [float(i)/normSum for i in totalVect]
        vectString = str(ex[0]) + ":" + ' '.join(str(x) for x in normed)

        fw.write(vectString + "\n")
    fw.close()
    