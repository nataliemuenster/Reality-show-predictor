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
					#balances data set a bit more
					if line[3] == 'publication' or line[3] == 'Vox' or line[3] == 'Washington Post' or line[3] == 'Reuters': #or line[2] == "Wonders of the universe":
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


def separateLabeledExamples(dataList): #could use classificationDict here instead
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

def filterStopWords(stopWords, words):
    """Filters stop words."""
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
        #c = list(b)
        c = [float(a[i]) + float(b[i]) for i in xrange(len(a))]
        #c[:len(a)] += a
        for j in xrange(len(a), len(b)):
	        c.append(float(b[j]))
        return c
    else:
        #c = list(a)
        #c[:len(b)] += b
        c = [float(a[i]) + float(b[i]) for i in xrange(len(b))]
        #c[:len(a)] += a
        for j in xrange(len(b), len(a)):
	        c.append(float(a[j]))
        return c

def incrementVectors(a, scale, b):
    #[weights[f] + v * self.eta for f, v in gradientLoss.items()]
    if len(a) < len(b):
        c = list(b)
        for i in xrange(len(c)):
            c[i] *= scale
        c[:len(a)] += a
        return c
    else:
        c = list(a)
        c[:len(b)] += b*scale
        return c

def vectorizeArticles(examples):
    wordVectDict = {}
    fr = open("../cs221-data/glove.6B.50d.txt", 'rb') 
    for line in fr:
            vectTerms = line.split()
            wordVectDict[vectTerms[0]] = vectTerms[1:]
            #print "WORD VECTOR FOR " + str(vectTerms[0]) + str(wordVectDict[vectTerms[0]])
    fr.close()
    #stopWords = readStopWordsFile()
    print "about to build article vectors"
    fw = open("article_word_vectors_debug.txt",'w')
    num = 0
    
    for ex in examples:
        num += 1
        if num % 100 == 0: print str(num) + " examples done!"
        totalVect = []
        words = ex[1]['text'].translate(None, string.punctuation).lower().split()
        #words = filterStopWords(stopWords, ex[1]['text'])
        
        for word in words:
            if word in wordVectDict:
                newVect = wordVectDict[word]
                totalVect = addVectors(newVect, totalVect)
                #print "TOTAL VECT combined words = " + str(totalVect)
        vectString = str(ex[0]) + ":" + ' '.join(str(x) for x in totalVect)
        #vectString = ' '.join(totalVect)
        fw.write(vectString + "\n")
    fw.close()
    