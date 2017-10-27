import collections
import math
import string

class NaiveBayes:
  def __init__(self):
    # maps class to list of [word count, doc count]
    self.classCounts = collections.defaultdict(lambda: [0, 0])
    # maps word --> class --> instances of word in class
    self.wordCounts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    # total number of docs
    self.nDocs = 0
  
  def classify(self, text):
    words = text.translate(None, string.punctuation).lower().split()
    klass = None
    maxProb = float("-inf")
    for c in self.classCounts:
      prior = math.log(self.classCounts[c][0]) - math.log(self.nDocs) #log of prior probability 
      prob = prior
      for word in words:
        if word in self.wordCounts: # skip over unseen words
          wordCount = self.wordCounts[word][c]
          #if wordCount == 0: wordCount = 1
          #print self.classCounts[c][1] + len(self.wordCounts)
          likelihood = math.log(wordCount + 1) - math.log(self.classCounts[c][1] + len(self.wordCounts)) 
          prob += likelihood
      if prob > maxProb: 
        maxProb = prob
        klass = c
    #returns None if there's an issue
    return klass

  #takes in text in doc and the doc's class
  def train(self, klass, text):
    words = text.translate(None, string.punctuation).lower().split()
    # number of appearances of word in docs with classification klass
    for word in words:
      self.wordCounts[word][klass] += 1
    # update total number of words in class klass
    self.classCounts[klass][0] += len(words)
    # number of docs with classification klass
    self.classCounts[klass][1] += 1
    # update total number of documents
    self.nDocs += 1