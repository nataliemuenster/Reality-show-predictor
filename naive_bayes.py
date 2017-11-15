import collections
import math
import string
import util

class NaiveBayes:
  def __init__(self):
    # maps class to list of [word count, doc count]
    #self.classCounts = collections.defaultdict(lambda: [0, 0])
    # maps word --> class --> instances of word in class
    #self.wordCounts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    
    self.wordCountsForClass = collections.defaultdict(lambda:collections.defaultdict(lambda: [0.0,0.0]))
    self.docCount = collections.defaultdict(lambda: 0.0)
    self.wordCounts = collections.defaultdict(lambda:[0.0,0.0])

    # total number of docs
    self.nDocs = 0.0
    self.vocab = set() #build vocabulary of unique words for pos and neg classes
    self.leftWords = [] #HARDCODE THESE
    self.rightWords = []
    self.stopWords = self.readStopWordsFile('./english.stop')

  def readStopWordsFile(self, fileName):
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    return ('\n'.join(contents)).split()

  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopWords and word.strip() != '':
        filtered.append(word)
    return filtered


  def nats_classify(self, text):
    words = text.translate(None, string.punctuation).lower().split()
    words = self.filterStopWords(words)
    uniqueWords = set()
    for word in words:
      uniqueWords.add(word)

    polarizingWords = set() #will this work??
    for word in uniqueWords:
        diff = math.fabs(self.wordCountsForClass[1][word][1] - self.wordCountsForClass[-1][word][1])
        #print diff, word, self.wordCountsForPosClass[word][bflag], self.wordCountsForNegClass[word][bflag]
        if diff > 55: #optimize this...? This number taken from 124
          polarizingWords.add(word)
    #print polarizingWords
    
    klass = -1
    leftCalc = math.log(self.docCount[-1] + 1)
    leftCalc -= math.log(self.docCount[1] + self.docCount[-1])
    rightCalc = math.log(self.docCount[1] + 1)
    rightCalc -= math.log(self.docCount[1] + self.docCount[-1])

    leftDenom = self.wordCounts[-1][1] + len(self.vocab) #3rd index 1 or 0??
    rightDenom = self.wordCounts[1][1] + len(self.vocab)

    for word in uniqueWords:
        multiplier = 1
        if (word in self.leftWords or word in self.rightWords):
          multiplier = 5
        #if (word in polarizingWords):
        #  multiplier = 4

        rightCalc += math.log((self.wordCountsForClass[1][word][0] + 1)**multiplier)
        rightCalc -= math.log(rightDenom)
        leftCalc += math.log((self.wordCountsForClass[-1][word][0] + 1)**multiplier)
        leftCalc -= math.log(leftDenom)

    if rightCalc > leftCalc:
      klass = 1

    return klass

  def classify(self, text):
    words = text.translate(None, string.punctuation).lower().split()
    klass = None
    maxProb = float("-inf")
    for c in self.classCounts:
      prior = math.log(self.classCounts[c][0]) - math.log(self.nDocs) #log of prior probability 
      prob = prior
      for word in words:
        if word in self.wordCounts: # skip over unseen words
          wordCount = self.wordCounts[word][c][1]
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
    words = self.filterStopWords(words)
    # number of appearances of word in docs with classification klass
    uniqueWords = set()
    for word in words:
      self.vocab.add(word) #build vocabulary of unique words for pos and neg classes
      uniqueWords.add(word)
      self.wordCounts[klass][0] += 1
      self.wordCountsForClass[klass][word][0] += 1

    self.docCount[klass] += 1
    self.wordCounts[klass][1] += len(uniqueWords)
    for uniq in uniqueWords:
      self.wordCountsForClass[klass][uniq][1] += 1

    self.nDocs += 1

