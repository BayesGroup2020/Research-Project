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


#trainingInstance = train.copy()
#trainingInstance = trainingInstance.drop(['fraud',"trustLevel", "totalScanTimeInSeconds", "trustLevel", "grandTotal", "lineItemVoids", "scansWithoutRegistration", "quantityModifications"],axis=1)

#"totalScanTimeInSeconds", "trustLevel", "grandTotal", "lineItemVoids", "scansWithoutRegistration", "quantityModifications"



class iForest:
    
    def __init__(self, data, numberOfTrees, maxHeight, subsamplingSize):#subsamplingSize=0.5(50% des Datensatzes)
     self.input = data
     self.numberOfInstances = len(data)
     self.numberOfTrees = numberOfTrees
     self.subsamplingSize = subsamplingSize
     self.heightLimit = maxHeight
     self.forest = list()
     self.createForest()
    
     
    

    def createForest(self):
     for x in range(self.numberOfTrees):
            sample = self.input.sample(frac=self.subsamplingSize, replace=True, random_state=1)
            newTree = Tree(sample,0,self.heightLimit)
            self.forest.append(newTree)
            return newTree
        

    
   
class Tree:
    
   def __init__(self, sample, currentTreeHeight, heightLimit):
    if currentTreeHeight == heightLimit or len(sample.index) <= 1 or sample["fraud"].nunique()==1:
        self.externalNode = True
        self.cardinalityX = len(sample)
        self.majorityClass = sample['fraud'].value_counts().argmax() # Als was soll klassifiziert werden?
    else:
        intervallPoints = self.getIntervallPoints(sample)
        
        splitVariable, splitTreshold = self.getbestFeatureTresholdForSplit(sample, intervallPoints)# getBestFeatureTresholdForSplit
        
        
        
        
        sampleToSplit = sample[splitVariable] # Zufälliges feature (ganze column) wählen
        
        
        self.splitAttribute = splitVariable # das zufällig gewählte feature
        self.splitValue = splitTreshold
        self.externalNode = False
    
        sampleLeft, sampleRight = self.filterBranches(sample)
        
        if (len(sampleLeft)==0 or len(sampleRight)==0):
            self.externalNode = True
            self.majorityClass = sample['fraud'].value_counts().argmax() 
        else:
            self.leftChild = Tree(sampleLeft, currentTreeHeight+1, heightLimit)
            self.rightChild = Tree(sampleRight, currentTreeHeight+1, heightLimit)
        
        
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
   def getIntervallPoints(self, bootstrapSample):
       sample1 = bootstrapSample.loc[:, bootstrapSample.columns != "fraud"]
       columnsToSample = sample1.sample(3, axis=1)
       
       dct = {}
       for i in range(3):
        currentFeature = columnsToSample.iloc[:, i].copy()
        currentFeature = currentFeature.to_frame()
        currentFeature["fraud"] = bootstrapSample["fraud"]
        currentFeature = currentFeature.sort_values(currentFeature.columns[0], ascending = True)
        dct[i] = []
        iteration = 0
        previousClass = currentFeature.iloc[0][1]
        for index, row in currentFeature.iterrows():
            #print(row[0],"askajskaasas")
            #print(currentFeature.iloc[iteration][0],"jsdksjkdksjd")
            currentClass = currentFeature.iloc[iteration][1]
            if(currentClass != previousClass):
                currentValue = currentFeature.iloc[iteration][0]
                previousValue = currentFeature.iloc[iteration-1][0]
                splitValue = (currentValue+previousValue)/2
                dct[i].append(splitValue)
            iteration+=1  
            
        dct = {k: list(set(v)) for k, v in dct.items()}  
        dct[columnsToSample.columns.values[i]] = dct.pop(i)  
        return dct
    
   def getbestFeatureTresholdForSplit(self, sample, intervallPoints):
        global LowestGiniIndex
        LowestGiniIndex = 100
        global finalSplitVariable
        global finalTreshold
        
        for key in intervallPoints.keys(): #reaching the keys of dict
            for value in intervallPoints[key]:
                # print(columnsToSample.columns[key], value)
                pSmaller = sample[key][sample[key]<=value].count()/len(sample)
                pLarger = sample[key][sample[key]>value].count()/len(sample)
        
                #print(sample[key])
                numberSmaller = sample[key][sample[key]<=value].count()
                numberLarger = sample[key][sample[key]>value].count()
                pSmallerAndFraud = len(sample[(sample[key]<=value) & (sample['fraud']==1)])/numberSmaller
                pSmallerAndNoFraud = len(sample[(sample[key]<=value) & (sample['fraud']==0)])/numberSmaller
                pLargerAndFraud = len(sample[(sample[key]>value) & (sample['fraud']==1)])/numberLarger
                pLargerAndNoFraud =len(sample[(sample[key]>value) & (sample['fraud']==0)])/numberLarger
                
                Gini1 = 1-(pSmallerAndFraud**2+pSmallerAndNoFraud**2)
                Gini2= 1-(pLargerAndFraud**2+pLargerAndNoFraud**2)
                GiniIndex = pSmaller*Gini1 + pLarger*Gini2
                if (GiniIndex < LowestGiniIndex):
                    LowestGiniIndex = GiniIndex
                    finalSplitVariable = key
                    finalTreshold = value
        return finalSplitVariable,finalTreshold

    
    
    
    
    
    
    
    
    
    
    
    
    
Forest = iForest(train,100,5,0.5)



#train['classification'] = train['anomalyScore'].apply(lambda x: 1 if x >= 0.9 else 0)
#train['correct'] = train['classification'] == train['fraud']
#correctPredictions = train['correct'] .values.sum() 
#falsePredictions = (~train['correct']).values.sum()

#accuracy = correctPredictions/(correctPredictions+falsePredictions)

#print(accuracy)
# print(correctPredictions)
# print(falsePredictions)
#print(train['correct'])
# print(train.loc[train["classification"]==1])


# Classificationtrain['MyColumn'].sum()

#incorrect = train.loc[train['correct'] == False]
#fraud = train.loc[train['fraud'] == 1]
#nofraud = train.loc[train['fraud'] == 0]


# Get outliers
# Q1 = train.quantile(0.25)
# Q3 = train.quantile(0.75)
# IQR = Q3 - Q1
# outliers = (train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))
# columnsWithOutliers = outliers.any();


# fraud predicted where fraud happended
#print(train.loc[(train["classification"]==1) & (train["correct"]==True)])
