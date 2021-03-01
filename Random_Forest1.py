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
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from numpy import arange
from numpy import argmax

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

################################ Random forest - basic classifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42, oob_score = True)
# Train the model on the training data
rf.fit(train, train_labels)
# Predict the train data
predictions_train = rf.predict(train)
print("Basic RF - training set")
print("accuracy scikit-learn:",metrics.accuracy_score(train_labels, predictions_train))
print("f1-score:", metrics.f1_score(train_labels, predictions_train))
print("Precision:", metrics.precision_score(train_labels, predictions_train))
print("Recall:", metrics.recall_score(train_labels, predictions_train))
print('Out-of-bag-Score: ', rf.oob_score_)
print("TotalCosts:", totalCosts(train_labels,predictions_train))

# Predict the test data
predictions_test = rf.predict(test)
print("Basic RF - test set")
print("accuracy scikit-learn:",metrics.accuracy_score(test_labels, predictions_test))
print("f1-score:", metrics.f1_score(test_labels, predictions_test))
print("Precision:", metrics.precision_score(test_labels, predictions_test))
print("Recall:", metrics.recall_score(test_labels, predictions_test))
print("TotalCosts:", totalCosts(test_labels, predictions_test))

print("--------------------------------------------")




# ########################################## Randomized search for best parameters
# Number of trees used for predictions
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum depth in tree
max_depth = [int(x) for x in np.linspace(2, 30, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2,4,]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2,4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# # Weights of classes
class_weight = ["balanced","balanced_subsample"]
# Criterion
criterion = ['gini','entropy']
#Out of bag score
oobs_score = [True]

# Create the grid
param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion':criterion,
                'oob_score':oobs_score,
                'class_weight':class_weight}
               
                
print(param_grid)

# Use the random grid to search for best hyperparameters
# base model to tune
rf_best = RandomForestClassifier()
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, use all available cores
rf_param = RandomizedSearchCV(estimator = rf_best, param_distributions = param_grid, n_iter = 100, cv = StratifiedKFold(n_splits=5), verbose=1, random_state=42, n_jobs = -1, scoring=totalCosts_scorer)
# Fit the random search model
rf_param.fit(train, train_labels)
best_param = rf_param.best_estimator_
# Print the best parameters, the resulting accuracy on the train and test set
print("The best params are:",rf_param.best_params_)## {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': None, 'bootstrap': False}
bestParameterRF = RandomForestClassifier(**rf_param.best_params_)
bestParameterRF.fit(train,train_labels)
predictions_train = bestParameterRF.predict(train)
# Predict the train data
predictions_train = rf.predict(train)
print("RF with optimized parameters - training set")
print("accuracy scikit-learn:",metrics.accuracy_score(train_labels, predictions_train))
print("f1-score:", metrics.f1_score(train_labels, predictions_train))
print("Precision:", metrics.precision_score(train_labels, predictions_train))
print("Recall:", metrics.recall_score(train_labels, predictions_train))
print('Out-of-bag-Score: ', bestParameterRF.oob_score_)
print("TotalCosts:", totalCosts(train_labels,predictions_train))

# Predict the test data
predictions_test = bestParameterRF.predict(test)
print("RF with optimized parameters - test set")
print("accuracy scikit-learn:",metrics.accuracy_score(test_labels, predictions_test))
print("f1-score:", metrics.f1_score(test_labels, predictions_test))
print("Precision:", metrics.precision_score(test_labels, predictions_test))
print("Recall:", metrics.recall_score(test_labels, predictions_test))
print("TotalCosts:", totalCosts(test_labels, predictions_test))


print("--------------------------------------------")










# ####################### Run the random forest with different class weights with scoring function precision
#Setting the range for class weights
# weights = np.linspace(0.0,0.99,200)
# #Creating a dictionary grid for grid search
# param_grid = {'lr__class_weight': [{0:x, 1:1.0-x} for x in weights]}
# #pipe = make_pipeline(StandardScaler(), LogisticRegression())
# pipe = Pipeline([("sc",StandardScaler()),("lr", RandomForestClassifier(n_estimators = 1000, random_state = 42))])
# #Fitting grid search to the train data with 5 fold stratified fold, focus on improving precision
# gridsearch = GridSearchCV(estimator= pipe, 
#                           param_grid= param_grid,
#                           cv=StratifiedKFold(), 
#                           n_jobs=-1, 
#                           scoring='precision', 
#                           verbose=1).fit(X_train, y_train)

# #### Ploting the score for different values of weights
# sns.set_style('whitegrid')
# plt.figure(figsize=(12,8))
# weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
# sns.lineplot(weigh_data['weight'], weigh_data['score'])
# plt.xlabel('Weight for class 2 (y=1)')
# plt.ylabel('Precision')
# plt.xticks([round(i/10,1) for i in range(0,11,1)])
# plt.title('Scoring for different class weights', fontsize=24)
# #### 
# # Get the best weights
# weight1,weight2 = list(gridsearch.best_params_.values())[0][0], list(gridsearch.best_params_.values())[0][1]
# # Make the prediction again with best weights
# pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 1000, random_state = 42, class_weight={0: weight1, 1: weight2}))
# pipe.fit(X_train, y_train)  # apply scaling on training data
# # Predict the validation data
# y_prediction_validation = pd.Series(pipe.predict(X_validation)) # apply the scaling on testing data and get prediction
# y_validation = y_validation.reset_index(drop=True)
# print("RF-grid search on weights, precision as scoring function - validation set")
# print("Accuracy:", metrics.accuracy_score(y_validation, y_prediction_validation))
# print("Precision:", metrics.precision_score(y_validation, y_prediction_validation))
# print("Recall:", metrics.recall_score(y_validation, y_prediction_validation))
# print("f1-score:", metrics.f1_score(y_validation, y_prediction_validation))
# print("TotalCosts:", totalCosts(y_validation, y_prediction_validation))

# # Die Accuracy, Precision, Recall und f1-Score für beide Klassen anzeigen lassen
# # print("Accuracy:", metrics.accuracy_score(y_validation, y_prediction_validation))
# # print("Precision:", metrics.precision_score(y_validation, y_prediction_validation,average=None))
# # print("Recall:", metrics.recall_score(y_validation, y_prediction_validation,average=None))
# # print("f1-score:", metrics.f1_score(y_validation, y_prediction_validation,average=None))

# #### Confusion matrix for prediction on validation set
# cnf_matrix = metrics.confusion_matrix(y_validation, y_prediction_validation)
# labels = [0, 1]
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(labels))
# plt.xticks(tick_marks, labels)
# plt.yticks(tick_marks, labels)
# # create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
# ax.xaxis.set_label_position("top")
# plt.title('Confusion matrix: Log. Regr. w best weights', y=1.1)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')

# # Predict the test data
# y_prediction_test = pd.Series(pipe.predict(test)) # apply the scaling on testing data and get prediction
# print("RF-grid search on weights, precision as scoring function - test set")
# print("Accuracy:", metrics.accuracy_score(test_labels, y_prediction_test))
# print("Precision:", metrics.precision_score(test_labels, y_prediction_test))
# print("Recall:", metrics.recall_score(test_labels, y_prediction_test))
# print("f1-score:", metrics.f1_score(test_labels, y_prediction_test))
# print("TotalCosts:", totalCosts(test_labels, y_prediction_test))




# ########################################## Run the random forest with different class weights with own scoring function
#
# #Setting the range for class weights
#weights = np.linspace(0.0,0.99,200)
#
# #### Run the gridsearch
#weights = np.linspace(0.0,0.99,200)
#
# #Creating a dictionary grid for grid search
#param_grid = {'lr__class_weight': [{0:x, 1:1.0-x} for x in weights]}
#
#
# ###### **rf_param.best_params_ Best parameters according to Randomized Search
# # 'oob_score': True,
# #  'n_estimators': 863,
# #  'min_samples_split': 2,
# #  'min_samples_leaf': 1,
# #  'max_features': 'auto',
# #  'max_depth': None,
# #  'criterion': 'entropy',
# #  'bootstrap': True
#
#
# #pipe = make_pipeline(StandardScaler(), LogisticRegression())
#pipe = Pipeline([("sc",StandardScaler()),("lr", RandomForestClassifier(oob_score = True,n_estimators=863,min_samples_split= 2, min_samples_leaf=1,max_features="auto",max_depth=None, criterion="entropy",bootstrap=True,random_state = 42))])
#
# #Fitting grid search to the train data with 5 fold stratified fold, focus on improving own cost function
#gridsearch = GridSearchCV(estimator= pipe, 
#                           param_grid= param_grid,
#                           cv=StratifiedKFold(), 
#                           n_jobs=-1, 
#                           scoring=totalCosts_scorer, 
#                           verbose=1).fit(train, train_labels)
#
# #### Ploting the score for different values of weight
#sns.set_style('whitegrid')
#plt.figure(figsize=(12,8))
#weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
#sns.lineplot(weigh_data['weight'], weigh_data['score'])
#plt.xlabel('Weight for class 2 (y=1)')
#plt.ylabel('TotalCosts')
#plt.xticks([round(i/10,1) for i in range(0,11,1)])
#plt.title('Scoring for different class weights', fontsize=24)
# #### 
#
# # Get the best weights
#weight1,weight2 = list(gridsearch.best_params_.values())[0][0], list(gridsearch.best_params_.values())[0][1]
# # Make the prediction again with best weights
#pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0: weight1, 1: weight2}))
#pipe.fit(X_train, y_train)  # apply scaling on training data
# # Make the prediction on the validation set
#y_prediction_validation = pd.Series(pipe.predict(X_validation)) # apply the scaling on testing data and get prediction
#y_validation = y_validation.reset_index(drop=True)
#print("RF-grid search on weights, own scoring function - validation set")
#print("Accuracy:", metrics.accuracy_score(y_validation, y_prediction_validation))
#print("Precision:", metrics.precision_score(y_validation, y_prediction_validation))
#print("Recall:", metrics.recall_score(y_validation, y_prediction_validation))
#print("f1-score:", metrics.f1_score(y_validation, y_prediction_validation))
#print("TotalCosts:", totalCosts(y_validation, y_prediction_validation))
#
# # Die Accuracy, Precision, Recall und f1-Score für beide Klassen anzeigen lassen
# # print("Accuracy:", metrics.accuracy_score(y_validation, y_prediction_validation))
# # print("Precision:", metrics.precision_score(y_validation, y_prediction_validation,average=None))
# # print("Recall:", metrics.recall_score(y_validation, y_prediction_validation,average=None))
# # print("f1-score:", metrics.f1_score(y_validation, y_prediction_validation,average=None))
#
# #### Confusion matrix for prediction on validation set
#cnf_matrix = metrics.confusion_matrix(y_validation, y_prediction_validation)
#labels = [0, 1]
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(labels))
#plt.xticks(tick_marks, labels)
#plt.yticks(tick_marks, labels)
# # create heatmap
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
#ax.xaxis.set_label_position("top")
#plt.title('Confusion matrix: Log. Regr. w best weights', y=1.1)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#
#
# # Predict the test data
#y_prediction_test = pd.Series(pipe.predict(test)) # apply the scaling on testing data and get prediction
#print("RF-grid search on weights, own scoring function - test set")
#print("Accuracy:", metrics.accuracy_score(test_labels, y_prediction_test))
#print("Precision:", metrics.precision_score(test_labels, y_prediction_test))
#print("Recall:", metrics.recall_score(test_labels, y_prediction_test))
#print("f1-score:", metrics.f1_score(test_labels, y_prediction_test))
#print("TotalCosts:", totalCosts(test_labels, y_prediction_test))
#
#
#################### Tuning classification threshold of last model
## apply threshold to positive probabilities to create labels
#def to_labels(pos_probs, threshold):
#	return (pos_probs >= threshold).astype('int')
## predict probabilities
#yhat = pipe.predict_proba(X_validation)
## keep probabilities for the positive outcome only
#probs = yhat[:, 1]
## define thresholds
#thresholds = arange(0, 1, 0.001)
## evaluate each threshold
#scores = [totalCosts(y_validation,to_labels(probs, t) ) for t in thresholds]
## get best threshold
#ix = argmax(scores)
#print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
#
##### Confusion matrix for prediction on validation set
#y_pred_best_threshold = (pipe.predict_proba(train)[:,1]>=thresholds[ix]).astype(int)
#
#
#cnf_matrix = metrics.confusion_matrix(train_labels, y_pred_best_threshold)
#labels = [0, 1]
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(labels))
#plt.xticks(tick_marks, labels)
#plt.yticks(tick_marks, labels)
## create heatmap
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
#ax.xaxis.set_label_position("top")
#plt.title('Confusion matrix: Random forest w best weights and threshold', y=1.1)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#



