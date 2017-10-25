from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import random
import collections
import math
import sys
import numpy as np


stopList = set(readFile('./english.stop'))


def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered




trainExamples = util.readExamples('???')
valExamples = util.readExamples('???')
testExamples = util.readExamples('???')

