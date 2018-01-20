import collections
import math
import string
import re
import time
import util
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class LinearClassifier:
    def __init__(self, numIters = 20, eta=.05):
        self.featureExtractor = self.extractWordFeatures
        self.numIters = numIters
        self.eta = eta
        self.sid = SentimentIntensityAnalyzer()
        self.features = self.readFeatureFile('./featureList.txt')

    def readFeatureFile(self, fileName):
        features = []
        f = open(fileName) 
        for line in f: #need to remove endline char from each line??
            keyword, regexes = line.split(', ')
            features.append((keyword, regexes[:-1])) #get rid of newline char
        f.close()
        return features #('\n'.join(contents)).split()


    #http://www.nltk.org/book/ch06.html
    #Cite SentiWordNet for positivity, negativity, objectivity (http://sentiwordnet.isti.cnr.it/)
    def extractWordFeatures(self, example):
        """
        Extract word features for a string x. Words are delimited by
        whitespace characters only.
        @param string x: 
        @return dict: feature vector representation of x.
        Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
        """
        featureDict = collections.defaultdict(float)
        # for sentence in example['text'].split('.'):
        #     ss = self.sid.polarity_scores(example['text'])
        #     featureDict["pos_sentiment"] += ss['pos']
        #     featureDict["neg_sentiment"] += ss['neg']
        #add parts of speech?   pos_tag(example)
        
        featureDict['title_sentiment'] = self.sid.polarity_scores(example['title'])['compound']
        #featureDict["title_sentiment"] = self.sid.polarity_scores(example['title'])['compound']
        #featureDict["total_sentiment"] = self.sid.polarity_scores(example['text'])['compound'] #takes too long, may not be important

        exampleText = example["text"].translate(None, string.punctuation).lower()
        exampleTitle = example["title"].translate(None, string.punctuation).lower()
        for feature in self.features:
            countText = len(re.findall(feature[1], exampleText))
            countTitle = len(re.findall(feature[1], exampleTitle))
            if countText > 0:
                featureDict[feature[0]] += countText
            if countTitle > 0:
                featureDict[feature[0]] += countTitle * 2 #optimize weighting here

        #SUPER SLOW with this... --> aded pronouns to featureList instead
        #get dif of stop words before this -- will it speed up???
        '''tags = pos_tag(example["text"])
        numAdj = 0
        numPron = 0
        for tag in tags: #(each in the form of [word, tag])
            if tag[1] == "PRON": numPron += 1
            elif tag[1] == "ADJ": numAdj += 1
        featureDict["pronouns"] = numPron
        featureDict["adjectives"] = numAdj
        '''

        # for word in example['text']:
        # 	# Political words, sentiment, etc, NOT every word. This helps us remove topical bias
        #     if word in featureDict:
        #         featureDict[word] += 1
        #     else:
        #         featureDict[word] = 1

        # Weight title words more?
        # for word in example['title']:
        # 	if word in featureDict:
        #         featureDict[word] += 10
        #     else:
        #         featureDict[word] = 10


        return featureDict

    def perform_sgd(self, trainExamples):
        weights = {}
        #print trainExamples[0]
        for i in range(self.numIters):
           # print "iteration " + str(i)
            for example in trainExamples:
                #phi(x)
                #print "Ex feature vectorized is: " + str(example[1])
                featureVector = self.featureExtractor(example[1])
                #y
                yValue = example[2]
                margin = 0
                for key in featureVector:
                    if weights.has_key(key):
                        margin += featureVector[key]*weights[key]
                margin *= yValue
                if margin < 1:
                    for key in featureVector:
                        if weights.has_key(key):
                            weights[key] = weights[key] + self.eta * yValue * featureVector[key]
                        else:
                            weights[key] = self.eta * yValue * featureVector[key]

                #calculates phi(x)y
                #for key in featureVector:
                #    featureVector[key] *= yValue

                #gradientLoss = {}
                #if w dot phi(x)y < 1, we use -phi(x)y (which cancels to just phi(x)y)
                #if util.dotProduct(weights, featureVector) < 1:
                #    gradientLoss = featureVector
                #util.increment(weights, self.eta, gradientLoss)
        return weights

    def classify(self, example, weights):
        if util.dotProduct(self.featureExtractor(example), weights) >= 0:
            return 1
        else:
            return -1

