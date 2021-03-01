#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jan-Niklas
"""


import pandas as pd
# graphics imports
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import math
from statistics import mean





# Import the data
train = pd.read_csv("Path to the training set",sep='|')
test  = pd.read_csv("Path to the test set",sep='|')
test_labels = pd.read_csv("Path to the real class labels",sep='|')
test['fraud'] = realclass["fraud"]
#train.head()


trainingInstance = train.copy()
trainingInstance = trainingInstance.drop(['fraud',"trustLevel", "totalScanTimeInSeconds", "trustLevel", "grandTotal", "lineItemVoids", "scansWithoutRegistration", "quantityModifications"],axis=1)



class iForest:
    
    def __init__(self, data, numberOfTrees, subsamplingSize):
     self.input = data
     self.numberOfInstances = len(data)
     self.numberOfTrees = numberOfTrees
     self.subsamplingSize = subsamplingSize
     self.heightLimit = math.ceil(math.log(subsamplingSize,2))
     self.forest = list()
     self.createForest()
     averageLengths = self.calculateAverageLength(self.forest, train)
     anomalyScores = self.calculateAnomalyScores(averageLengths)
     #print(anomalyScores)
     j2 = [i for i in anomalyScores if i > 0.4]
     
     train['anomalyScore'] = anomalyScores
     
     #allElementsFromTrainWhichAreFraud = train.loc[train["fraud"]==1]
     
    

    def createForest(self):
     for x in range(self.numberOfTrees):
            sample = self.input.sample(n=self.subsamplingSize)
            newTree = iTree(sample,0,self.heightLimit)
            self.forest.append(newTree)
            return newTree
        

    def calculateAverageLength(self, forest, trainingInstances):   
    
     self.averageLengths = list()
        
     for index, row in trainingInstances.iterrows():
        lengths = list()
        for iTree in forest:
            length = self.pathLength(iTree, row, 0)
            lengths.append(length)
        self.averageLengths.append(mean(lengths))
     return self.averageLengths

    def pathLength(self, iTree, trainingInstance, currentPathLength):
     if (iTree.isExternalNode()):
           size = iTree.getSize()
           if (size > 1):
            additionalTerm = self.calculateAveragePathLengthUnsuccessfulSearch(size)
            return currentPathLength + additionalTerm
           else:
            return currentPathLength
     splitAttribute, splitValue = iTree.getSplitAttributeValue()
     leftChild, rightChild = iTree.getChildren()
     
     #print(splitAttribute, splitValue)
     #print(trainingInstance[splitAttribute].values[0])
     if trainingInstance[splitAttribute].values[0] < splitValue:
         return self.pathLength(leftChild, trainingInstance, currentPathLength+1)
     else:
         return self.pathLength(rightChild, trainingInstance, currentPathLength+1)
      
    def calculateAveragePathLengthUnsuccessfulSearch(self, size):
         n = size
         return 2 * (np.log(n-1)+0.5772156649)-(2*(n-1)/(n))
     
    def calculateAnomalyScores(self, averageLengthsList):
        anomalyScores = list()
        averageLengthUnsuccessfulSearch = self.calculateAveragePathLengthUnsuccessfulSearch(self.numberOfInstances)
        for averageLenght in averageLengthsList:
            anomalyScore = 2**(-averageLenght/averageLengthUnsuccessfulSearch)
            anomalyScores.append(anomalyScore)
        return anomalyScores
class iTree:
    
   def __init__(self, sample, currentTreeHeight, heightLimit):
    if currentTreeHeight > heightLimit or len(sample.index) <= 1:
        self.externalNode = True
        self.cardinalityX = len(sample)
    else:
        sampleToSplit = sample.sample(axis=1) # Zufälliges feature (ganze column) wählen
        self.splitAttribute = list(sampleToSplit) # das zufällig gewählte feature
        self.splitAttributeMin, self.splitAttributeMax = list(sampleToSplit.min(axis=0)), list(sampleToSplit.max(axis=0))
        self.splitValue = np.random.uniform(self.splitAttributeMin, self.splitAttributeMax)
        self.externalNode = False
    
        sampleLeft, sampleRight = self.filterBranches(sample)
        
        # print(sampleLeft)
        # print(sampleRight)
 
        self.leftChild = iTree(sampleLeft, currentTreeHeight+1, heightLimit)
        self.rightChild = iTree(sampleRight, currentTreeHeight+1, heightLimit)
        
        
   def filterBranches (self, sample):
        sample = sample.copy()
        isSmaller = (sample[self.splitAttribute] < self.splitValue).squeeze()
        isLarger = (sample[self.splitAttribute] >= self.splitValue).squeeze()
        sampleLeft = sample[isSmaller]
        sampleRight = sample[isLarger]
        return sampleLeft, sampleRight
    
   def getSplitAttributeValue(self):
       return self.splitAttribute, self.splitValue
   def getChildren(self):
       return self.leftChild, self.rightChild
   def isExternalNode(self):
       return self.externalNode
   def getSize(self):
       return self.cardinalityX
    
Forest = iForest(trainingInstance,100,256)



train['classification'] = train['anomalyScore'].apply(lambda x: 1 if x >= 0.9 else 0)
train['correct'] = train['classification'] == train['fraud']
correctPredictions = train['correct'] .values.sum() 
falsePredictions = (~train['correct']).values.sum()

accuracy = correctPredictions/(correctPredictions+falsePredictions)

print(accuracy)
# print(correctPredictions)
# print(falsePredictions)
#print(train['correct'])
# print(train.loc[train["classification"]==1])


# Classificationtrain['MyColumn'].sum()

incorrect = train.loc[train['correct'] == False]
fraud = train.loc[train['fraud'] == 1]
nofraud = train.loc[train['fraud'] == 0]


# Get outliers
Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1
outliers = (train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))
columnsWithOutliers = outliers.any();


# fraud predicted where fraud happended
#print(train.loc[(train["classification"]==1) & (train["correct"]==True)])


