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
        #zero-fills?
        if len(a) < len(b):
            c = b.copy()
            c[:len(a)] += a
            return c
        else:
            c = a.copy()
            c[:len(b)] += b
            return c

def writePretrainedWordVectorsToFile():
    wordVectDict = {}
    fr = open("../cs221-data/glove.42B.300d.txt", 'rb') 
    for line in fr:
            vectTerms = line.split()
            wordVectDict[vectTerms[0]] = vectTerms[1:]
    fr.close()
    print "Huge dict object created"
    with open("pretrained_word_vector_dict2.txt",'wb') as fw:
        pkl.dump(wordVectDict, fw)
    fw.close()
    return wordVectDict

def vectorizeArticles2(examples, wordVectDict):
    with open("article_word_vectors2.txt", 'wb') as f:
        for ex in examples:
            totalVect = np.array([])
            for word in ex.text:
                newVect = np.array(wordVectDict[word])
                totalVect = util.addVectors(newVect, totalVect)
            vect = np.linalg.norm(totalVect).toList()
            pkl.dump(vect, f)
            print "Article added!"
    file.close()

def vectorizeArticles(examples):
    wordVectDict = {}
    fr = open("../cs221-data/glove.42B.300d.txt", 'rb') 
    for line in fr:
            vectTerms = line.split()
            wordVectDict[vectTerms[0]] = vectTerms[1:]
    fr.close()

    stopWords = util.readStopWordsFile()

    fw = open("article_word_vectors.txt",'w')
    num = 0
    print "about to build article vectors"
    for ex in examples:
        num += 1
        if num % 100 == 0: print str(num) + " examples done!"
        totalVect = np.array([])
        words = util.filterStopWords(stopWords, ex.txt)
        for word in words:
            newVect = np.array(wordVectDict[word])
            totalVect = util.addVectors(newVect, totalVect)
        vect = np.linalg.norm(totalVect).toList()
        fw.write(vect)
    fw.close()

