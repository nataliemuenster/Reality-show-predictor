import collections
import math
import string
import naive_bayes as nb

class SemiSupervised:
  def __init__(self, function, labeledList):
    # maps class to list of [word count, doc count]
    self.function = function
    self.labeled = labeledList

  def addData(self, dataPoint, label):
    self.labeled.append((dataPoint, label))

  def predict(self, text):
    return self.function.classify(text)

  #takes in text in doc and the doc's class
  def train(self):
    for data, klass in self.labeled:
      self.function.train(klass, data['text'])