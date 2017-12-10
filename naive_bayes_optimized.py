import collections
import math
import string
import util
import re
#import sgd
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
    self.stopWords = self.readStopWordsFile('./english.stop')
    self.n = 1
    self.sgd_feature_list = self.readFeatureFile('./featureList.txt')
    self.sgd_feature_names = self.getFeatureNames()
    self.SGD_features = True
  
  def readFeatureFile(self, fileName):
    features = []
    f = open(fileName) 
    for line in f: #need to remove endline char from each line??
        keyword, regexes = line.split(', ')
        features.append((keyword, regexes[:-1])) #get rid of newline char
    f.close()
    return features #('\n'.join(contents)).split()

  def getFeatureNames(self):
    featureNames = set()
    for feature in self.sgd_feature_list:
      featureNames.add(feature[0])
    return featureNames

  def setN(self, n):
    print "setting n to %r" % n
    self.n = n

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


  def classify(self, text, isUnsupervised=False):
    uniqueWords = set()
    if isUnsupervised:
      uniqueWords = text
    else:
      #words = text.translate(None, string.punctuation).lower().split()
      #words = self.filterStopWords(words)
      words = self.get_ngrams(text)
      uniqueWords = set()
      for word in words:
        uniqueWords.add(word)

    polarizingWords = set() #will this work??
    for word in uniqueWords:
        diff = math.fabs(self.wordCountsForClass[1][word][0] - self.wordCountsForClass[-1][word][0])
        #print diff, word, self.wordCountsForClass[1][word][0], self.wordCountsForClass[-1][word][0]
        if diff > 40:
          polarizingWords.add(word)
    klass = -1
    leftCalc = math.log(self.docCount[-1] + 1)
    leftCalc -= math.log(self.docCount[1] + self.docCount[-1])
    rightCalc = math.log(self.docCount[1] + 1)
    rightCalc -= math.log(self.docCount[1] + self.docCount[-1])

    leftDenom = self.wordCounts[-1][1] + len(self.vocab) #3rd index 1 or 0??
    rightDenom = self.wordCounts[1][1] + len(self.vocab)

    for word in uniqueWords:
        diff = math.fabs(self.wordCountsForClass[1][word][0] - self.wordCountsForClass[-1][word][0])
        multiplier = 0
        if diff <= (self.nDocs * .01): multiplier = .01
        if diff > (self.nDocs * .01): multiplier = math.log(diff)
        if (word in self.sgd_feature_names): multiplier *= 3 #tune this?

        #multiplier = 1
        #changed from word in polarizing words w. multiplier 3
        #if (word in polarizingWords):
        #  multiplier *= 2
          #diff = math.fabs(self.wordCountsForClass[1][word][0] - self.wordCountsForClass[-1][word][0])
          #multiplier = math.log(diff)
          #print "multiplier: %r" % multiplier

        rightCalc += math.log((self.wordCountsForClass[1][word][1] + 1)**multiplier)
        rightCalc -= math.log(rightDenom)
        leftCalc += math.log((self.wordCountsForClass[-1][word][1] + 1)**multiplier)
        leftCalc -= math.log(leftDenom)

    if rightCalc > leftCalc:
      klass = 1

    return klass

  def get_ngrams(self, text):
    words = []
    if self.n > 1: text = 'sentenceend' + re.sub('[!.?]', ' sentenceend ', text) + 'sentenceend'
    trimmedText = text.translate(None, string.punctuation).lower().split()
    trimmedText = self.filterStopWords(trimmedText)
    #if self.n == 1: (the way it is now, it combines unigrams with ngrams)
    words = list(trimmedText)
    if self.n > 1: 
      for i in range (0, len(trimmedText) - self.n):
        ngram = []
        for j in range(0, self.n):
          ngram.append(trimmedText[i + j])
        words.append(' '.join(ngram))
    return words

  def extract_SGD_features(self, text):
    text = text.translate(None, string.punctuation).lower()
    sgd_features = []
    for feature in self.sgd_feature_list:
      countText = len(re.findall(feature[1], text))
      if countText > 0:
        sgd_features.append(feature[0])
    return sgd_features

  #takes in text in doc and the doc's class
  def train(self, klass, text, isUnsupervised=False):
    if isUnsupervised:
      self.wordCounts[klass][1] += len(text)
      self.docCount[klass] += 1
      for uniq in text:
        self.wordCountsForClass[klass][uniq][1] += 1
        self.vocab.add(uniq)
    else:
      words = self.get_ngrams(text)
    if self.SGD_features:
      sgd_features = self.extract_SGD_features(text)
      words = words + self.extract_SGD_features(text)
      # number of appearances of n-gram in docs with classification klass
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

