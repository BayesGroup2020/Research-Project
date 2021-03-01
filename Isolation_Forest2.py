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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

############################# Import the data
train = pd.read_csv("Path to the training set",sep='|')
test  = pd.read_csv("Path to the test set",sep='|')
test_labels = pd.read_csv("Path to the real class labels",sep='|')
test['fraud'] = test_labels["fraud"]
# Save labels 
train_labels = train["fraud"]
test_labels = test["fraud"]
# Remove labels from the features
train= train.drop('fraud', axis = 1)
test= test.drop('fraud', axis = 1)
# Feature names
feature_list = list(train.columns)
# Convert to numpy array
#train = np.array(train)
#### Create training and validation set from train 
# Name train dataset as X and fraud/no fraud as y
y = train_labels.to_frame().fraud.copy()
X = train.copy()
# Split the training set (100%) into train (75%) and validation set (25%)
X_train, X_validation, y_train, y_validation= train_test_split(X, y, test_size=0.25, random_state=123)

################################ Create a cost function for evaluation
# Own scoring function
def totalCosts(y_true, y_prediction):
    cnf_matrix = metrics.confusion_matrix(y_true, y_prediction)
    TN = cnf_matrix[0][0]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[1][1]
    FP = cnf_matrix[0][1]
    return (TP*5-FN*(5)-25*FP)

# Create the scorer
totalCosts_scorer = make_scorer(totalCosts, greater_is_better=True)

################################ Isolation forest
clf=IsolationForest(n_estimators=150, max_samples='auto', contamination='auto',max_features=1.0)
train_labels = train_labels.to_frame()
clf.fit(train)

# predictions
y_pred_train = clf.predict(train)
y_pred_test = clf.predict(test)

# Reshape the prediction values to 0 for valid, 1 for fraud. 
y_pred_train[y_pred_train == 1] = 0
y_pred_train[y_pred_train == -1] = 1
y_pred_test[y_pred_test == 1] = 0
y_pred_test[y_pred_test == -1] = 1

#evaluation of the model
#printing every score of the classifier
from sklearn.metrics import confusion_matrix
#n_outliers = len(Fraud)
acc_train= metrics.accuracy_score(train_labels,y_pred_train)
acc_test= metrics.accuracy_score(test_labels,y_pred_test)
print("The accuracy is {}".format(acc_train))
print("The accuracy is {}".format(acc_test))

print("LR-grid search on weights, precision as scoring function - training set")
print("Accuracy:", metrics.accuracy_score(test_labels,y_pred_test))
print("Precision:", metrics.precision_score(test_labels,y_pred_test))
print("Recall:", metrics.recall_score(test_labels,y_pred_test))
print("f1-score:", metrics.f1_score(test_labels,y_pred_test))
print("TotalCosts:", totalCosts(test_labels,y_pred_test))

print(totalCosts(train_labels,y_pred_train))
print(totalCosts(test_labels,y_pred_test))
# prec= precision_score(Y_test,y_pred)
# print("The precision is {}".format(prec))
# rec= recall_score(Y_test,y_pred)
# print("The recall is {}".format(rec))
# f1= f1_score(Y_test,y_pred)
# print("The F1-Score is {}".format(f1))
# MCC=matthews_corrcoef(Y_test,y_pred)
# print("The Matthews correlation coefficient is{}".format(MCC))

#################################