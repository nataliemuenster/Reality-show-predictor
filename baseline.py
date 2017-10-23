from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import random
import collections
import math
import sys
import numpy as np


ngram_threshold = 7

#extract features, fit regression
def fitModel(examples, vocab=None, frequent_ngram_col_idx=None):
	corpus = [] #each piece of data, without its classification
        for x,y in examples:
            corpus.append(x)
            print x
        vectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(1, 3),token_pattern=r'\b\w+\b', min_df=1)
        X = vectorizer.fit_transform(corpus)
        
        # analyze = vectorizer.build_analyzer()
        fullfeature = X.toarray()

        print 'SHAPE', len(fullfeature), len(fullfeature[0])

        # The most time expensive part (pruning so only frequent ngrams used)
        '''
        if not frequent_ngram_col_idx:
            frequent_ngram_col_idx = []
            for i in range(fullfeature.shape[1]):
                if sum(fullfeature[:,i]) > ngram_threshold:
                    frequent_ngram_col_idx.append(i)
        fullfeature = fullfeature[:, frequent_ngram_col_idx]
        print 'NEW SHAPE', len(fullfeature), len(fullfeature[0])
        '''
        #Add features from grammatical context in transcript

        fullfeature = contextualFeatures(examples, fullfeature)

        print 'CONTEXTUAL SHAPE', len(fullfeature), len(fullfeature[0])
        # return vectorizer
        return fullfeature, vectorizer.vocabulary_, frequent_ngram_col_idx





def contextualFeatures(examples):
	add_features = np.zeros((len(fullfeature), 8)) #how many new features to add?

	for line in xrange(len(examples), fullfeature): 
        #add features here

	fullfeature = np.hstack((fullfeature, add_features))
    return fullfeature

def learnPredictor(trainExamples, devExamples, testExamples):
	print 'BEGIN: GENERATE TRAIN'
    trainFeatures, vocabulary, freq_col_idx = fitModel(trainExamples)
    trainX = trainFeatures
	trainY = [y for x,y in trainExamples]
	print 'END: GENERATE TRAIN'
	
    print 'BEGIN: GENERATE DEV'
    devFeatures, _, freq_col_idx_dev = fitModel(devExamples, vocab=vocabulary, frequent_ngram_col_idx=freq_col_idx)
    devX = devFeatures
    devY = [y for x,y in devExamples]
    print 'END: GENERATE DEV'
    
    print 'BEGIN: GENERATE TEST'
    testFeatures, _, freq_col_idx_test = fitModel(testExamples, vocab=vocabulary, frequent_ngram_col_idx=freq_col_idx)
    testX = testFeatures
    testY = [y for x,y in testExamples]
	print 'END: GENERATE TEST'
    
    print "BEGIN: TRAINING"
    regr = LogisticRegression()
    regr.fit(trainX, trainY)
    print "END: TRAINING"
    trainPredict = regr.predict(trainX)
    devPredict = regr.predict(devX)
    testPredict = regr.predict(testX)
    precision,recall,fscore,support = precision_recall_fscore_support(trainY, trainPredict, average='binary')
    print "LOGISTIC TRAIN scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
    
    precision,recall,fscore,support = precision_recall_fscore_support(devY, devPredict, average='binary')
    print "LOGISTIC DEV scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)

    precision,recall,fscore,support = precision_recall_fscore_support(testY, testPredict, average='binary')
    print "LOGISTIC TEST scores:\n\tPrecision:%f\n\tRecall:%f\n\tF1:%f" % (precision, recall, fscore)
    return vocabulary, freq_col_idx, regr





trainExamples = util.readExamples('switchboardsampleL.train')
valExamples = util.readExamples('switchboardsampleL.val')
testExamples = util.readExamples('switchboardsampleL.test')
vocabulary, freq_col_idx, regr = learnPredictor(trainExamples, valExamples, testExamples)

