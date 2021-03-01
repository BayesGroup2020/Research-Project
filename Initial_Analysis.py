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



# Import the data
train = pd.read_csv("Path to the training set",sep='|')
test  = pd.read_csv("Path to the test set",sep='|')
test_labels = pd.read_csv("Path to the real class labels",sep='|')
test['fraud'] = realclass["fraud"]
#train.head()



# Inspect the columns and their datatypes
print(train.info())


### Boxplots
sns.boxplot(x=train["totalScanTimeInSeconds"])
sns.boxplot(x=train["grandTotal"])
sns.boxplot(x=train["lineItemVoids"])
sns.boxplot(x=train["scansWithoutRegistration"])
sns.boxplot(x=train["quantityModifications"])
sns.boxplot(x=train["scannedLineItemsPerSecond"])
sns.boxplot(x=train["valuePerSecond"])
sns.boxplot(x=train["lineItemVoidsPerPosition"])


# Whenever fraud is happening, what is the trust level?
pd.crosstab(train['fraud'], train['trustLevel'])

# How many scans without registration are there, when fraud is happening?
pd.crosstab(train['fraud'], train['scansWithoutRegistration'])





# Checking the means of the classes fraud and no fraud for different columns

# fraud = train.loc[train['fraud'] == 1]
# nofraud = train.loc[train['fraud'] == 0]

# fraud["totalScanTimeInSeconds"].mean()
# fraud["scansWithoutRegistration"].mean()
# fraud["lineItemVoidsPerPosition"].mean()
# nofraud["totalScanTimeInSeconds"].mean()
# nofraud["scansWithoutRegistration"].mean()
# nofraud["lineItemVoidsPerPosition"].mean()





### Outlier detection

# For the outlier detection we use the z-score and define a treshold to exclude outliers.
# The output are two arrays, the first indicating the row of the outlier, the second indicating the 
# column of the outlier. The first example is the 49 row and 5 column with a z-score of 5.18.

z = np.abs(stats.zscore(train))
print(np.where(z > 3))
# print(z[49][5])


# If we then exclude all entries with a z-score > 3, we get a new dataframe. 
# This dataframe has only 1708 entries (1879 entries initially). These entries are all 
# classified as no fraud. Initially the dataset included 104 entries classified as fraud.

#train = train[(z < 3).all(axis=1)]
print(train.info())
train.fraud.value_counts()

# Changing the datatype of trustLevel and fraud to category
train['trustLevel'] = train.trustLevel.astype('category')
train['fraud'] = train.fraud.astype('category')


### Plots
# Compare the count of fraud and no fraud in the training set
fraudPlot = sns.countplot(x="fraud", data=train, palette=["green","red"])
fraudPlot.set(title='Training set: no fraud vs. fraud', xlabel='Fraud', ylabel='Count')


print(test.info())

# Compare the number of entries in training set and test set: 
# Training: 1879 entries
# Testing: 498121 entries
df = pd.DataFrame({'Dataset':['Training', 'Test'], 'Number of entries':[1879, 498121]})
ax = df.plot.bar(x='Dataset', y='Number of entries', rot=0)



### Compute the baseline: 
# traing set: no fraud vs. fraud: 1775 vs. 104 entries
train.fraud.value_counts()
baseline_train = 1775 / (len(train.index)) 
train_loss = 104*(-5)

# test set: no fraud vs. fraud: 474394 vs. 23727 entries
test.fraud.value_counts()
baseline_test = test.fraud.value_counts()[0] / (len(test.index)) 
test_loss = 23727*(-5)



### Create a decision tree 


# # Create the classifier based on the train dataset
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
# clf.fit(train.drop(['fraud'], axis=1), train['fraud'])

# # Make a prediction for the test dataset
# clf.predict(test.drop(['fraud'], axis=1))

# # The score method returns the accuracy of the model
# score = clf.score(test.drop(['fraud'], axis=1), test["fraud"])
# print(score)

# # Tune parameters of decision tree for better results
# # List of values to try for max_depth:
# max_depth_range = list(range(1, 10))
# # List to store the average RMSE for each value of max_depth:
# accuracy = []
# for depth in max_depth_range:
    
#     clf = DecisionTreeClassifier(max_depth = depth, 
#                              random_state = 0)
#     clf.fit(train.drop(['fraud'], axis=1), train['fraud'])
#     score = clf.score(test.drop(['fraud'], axis=1), test["fraud"])
#     accuracy.append(score)

# # Plot accuracy depending on max_depth_range
# plt.plot(max_depth_range,accuracy)



# # Weitere Plots

# df.hist(column='session_duration_seconds')


# ax.scatter(train["fraud"], train["trustLevel"])
# ax.set_xlabel("HP")
# ax.set_ylabel("Price")
# plt.show()




# SMOTE TEST

X = train.copy()
X = X.drop('fraud', 1)
X = X.values
Y = train[['fraud']].copy().values
Y = Y.reshape(1879,)

from collections import Counter
from imblearn.over_sampling import SMOTE 
from sklearn.datasets import make_classification
print('Original dataset shape %s' % Counter(Y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, Y)
print('Resampled dataset shape %s' % Counter(y_res))

#X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)




# # Create the classifier based on the train dataset
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
# clf.fit(train.drop(['fraud'], axis=1), train['fraud'])

# # Make a prediction for the test dataset
# clf.predict(test.drop(['fraud'], axis=1))

# # The score method returns the accuracy of the model
# score = clf.score(test.drop(['fraud'], axis=1), test["fraud"])
# print(score)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
lf = clf.fit(X, Y)

#k = clf.predict(test.drop(['fraud'], axis=1))

# The score method returns the accuracy of the model
score = lf.score(test.drop(['fraud'], axis=1), test["fraud"])
print(score)



