import collections
import math
import string
import re
import random
import sys
import numpy as np
import os
import csv
import unsupervised as us
import util
import time 

class Kmeans:
    def __init__(self, k = 2, maxIters = 10):
        '''K: number of desired clusters. Assume that 0 < K <= |examples|.
        maxIters: maximum number of iterations to run, terminates early upon convergence.
        '''
        self.maxIters = maxIters
        self.examples = []
        self.examplesSQD = []
        self.K = k
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

    def createExampleVector(self, example):
        words = example["text"].translate(None, string.punctuation).lower().split()
        words = self.filterStopWords(words)
        wordCounts = collections.defaultdict(int) #unigram model to represent each article
        for word in words:
            wordCounts[word] += 1
        self.examples.append(wordCounts)
        
        sq = 0
        for key, val in wordCounts.items():
            sq += val**2
        self.examplesSQD.append(sq)

    def findClosestCluster(self, centroidsSQD, centroids, ex):
        distances = []
        for c in xrange(len(centroids)):
            dist = 0.0; 
            dist += self.examplesSQD[ex]
            dist += centroidsSQD[c]
            distances.append(dist - (2 * util.dotProduct(self.examples[ex], centroids[c])))

        return distances.index(min(distances))

    def runKmeans(self):
        '''
        examples: list of examples, each example is a dict representing a sparse vector.
        Return: (length K list of cluster centroids,
                list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
                final reconstruction loss)
        '''
        #randomly choose start centroids as different random example vectors
        centroids = []
        #generate initial centroid locations
        for k in xrange(self.K):
            randEx = random.randint(0, len(self.examples)-1)
            #print "RandEx:" + str(randEx)
            randCentroid = self.examples[randEx]
            
            while randCentroid in centroids: #sample without replacement
                randEx = np.random.randint(0, len(self.examples)-1)
                randCentroid = self.examples[randEx]
                #print "hi -- duplicate randint, now = " + str(randEx)
            #print "rand centroid" + str(randCentroid)
            centroids.append(randCentroid)
        #print "done"
        
        #precompute example^2
        '''examplesSQD = []
        for ex in xrange(len(self.examples)):
            sq = 0
            for key, val in self.examples[ex].items():
                sq += val**2
            examplesSQD.append(sq)'''
            
        for iters in xrange(self.maxIters):
            #print "iteration " + str(iters)
            #print "CLUSTERSSSSS: " + str(centroids)
            #precompute centroid^2
            centroidsSQD = []
            for c in xrange(len(centroids)):
                csq = 0
                for key, val in centroids[c].items():
                    csq += val**2
                centroidsSQD.append(csq)

            clusterAssignments = [0] * len(self.examples) #each example mapped to cluster it belongs to
            
            for pt in xrange(len(self.examples)):
                closest = self.findClosestCluster(centroidsSQD, centroids, pt)
                clusterAssignments[pt] = closest
            #print "examples assigned, now recalculating clusters. (assignments): " + str(clusterAssignments)
            #Compute the new centroids as the average of all features assigned to it
            clusterSize = [0] * self.K
            clusterContents = [{} for h in xrange(self.K)]
            #print "EMPTY CLUSTERS?" + str(clusterContents)
            for j in xrange(len(self.examples)):
                util.increment(clusterContents[clusterAssignments[j]], 1, self.examples[j])
                clusterSize[clusterAssignments[j]] += 1
            #print "CLUSTER CONTENTS: " + str(clusterContents)
            newCentroids = []
            for i in xrange(len(centroids)):
                newC = {}

                util.increment(newC, 1.0/clusterSize[i], clusterContents[i])
                newCentroids.append(newC)
            
            if newCentroids == centroids:
                #print "converged"
                break
            centroids = newCentroids

        reconstrLoss = 0.0
        '''for x in xrange(len(self.examples)):
            reconstrLoss += self.examplesSQD[x]
            reconstrLoss += centroidsSQD[clusterAssignments[x]]
            reconstrLoss -= 2 * util.dotProduct(self.examples[x], centroids[clusterAssignments[x]])'''
        
        return centroids, clusterAssignments, reconstrLoss

