import collections
import math
import string

class SGD:
	def __init__(self, numIters = 20, eta=.01):
		self.featureExtractor = self.extractWordFeatures
		self.numIters = numIters
		self.eta = eta

	def extractWordFeatures(self, example):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    listOfWords = {'????????'}
    featureDict = {}
    for word in example['text']:
    	# Political words, sentiment, etc, NOT every word. This helps us remove topical bias
    	if not word in listOfWords:
            continue
        if word in featureDict:
            featureDict[word] += 1
        else:
            featureDict[word] = 1

    # Weight title words more?
    for word in example['title']:
    	if word in featureDict:
            featureDict[word] += 10
        else:
            featureDict[word] = 10
    return featureDict


	def dotProduct(self, d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())


	def increment(self, d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


	def perform_sgd(self, trainExamples):
		weghts = {}
		for i in range(self.numIters):
	        for example in trainExamples:
	            #phi(x)
	            featureVector = self.featureExtractor(example[0])
	            #y
	            yValue = example[1]
	            #calculates phi(x)y
	            for key in featureVector:
	                featureVector[key] *= yValue

	            gradientLoss = {}
	            #if w dot phi(x)y < 1, we use -phi(x)y (which cancels to just phi(x)y)
	            if self.dotProduct(weights, featureVector) < 1:
	                gradientLoss = featureVector
	            self.increment(self.weights, self.eta, gradientLoss)
	    return weights

	def predict(self, example, weights):
        if self.dotProduct(self.featureExtractor(example), weights) >= 0:
            return 1
        else:
            return -1