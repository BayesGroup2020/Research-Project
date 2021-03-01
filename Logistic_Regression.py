#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jan-Niklas
"""
import pandas as pd
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
from sklearn.linear_model import LogisticRegression
from numpy import arange
from numpy import argmax


from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Input, Dropout
from keras import Sequential
from keras import backend as K
from keras.regularizers import l2
from sklearn.metrics import classification_report

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
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[1][1]
    FP = cnf_matrix[0][1]
    return (5*FN+25*FP-5*TP)
# Create the scorer
totalCosts_scorer = make_scorer(totalCosts, greater_is_better=True)

# #Logistic regression model with Keras - with binary crossentropy
#model = Sequential()
#model.add(Dense(1, input_shape=(9,), activation='sigmoid', kernel_regularizer=l2(0.)))
#model.compile(optimizer='sgd', loss='binary_crossentropy')
##model.fit(scaled_train, y_train, nb_epoch=100)
#
## Create a Pipeline to scale the data and make a prediction on the validation set
#pipe = make_pipeline(StandardScaler(), model)
#pipe.fit(train, train_labels, sequential__nb_epoch=100)
#y_prediction_validation = pd.Series(pipe.predict(X_validation).ravel())
#y_prediction_validation = (y_prediction_validation>0.5)
#
#
##y_prediction_validation = pd.Series(pipe.predict(train)) # apply the scaling on validation data and get prediction
#y_validation = y_validation.reset_index(drop=True)
#print("Basic LR - validation set")
#print("Accuracy:", metrics.accuracy_score(y_validation, y_prediction_validation))
#print("Precision:", metrics.precision_score(y_validation, y_prediction_validation))
#print("Recall:", metrics.recall_score(y_validation, y_prediction_validation))
#print("f1-score:", metrics.f1_score(y_validation, y_prediction_validation))
#print("TotalCosts:", totalCosts(y_validation, y_prediction_validation))

## Create a confusion matrix
#cnf_matrix = metrics.confusion_matrix(y_validation, y_prediction_validation)
#labels = [0, 1]
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(labels))
#plt.xticks(tick_marks, labels)
#plt.yticks(tick_marks, labels)
## create heatmap
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
#ax.xaxis.set_label_position("top")
#plt.title('Confusion matrix: Log. Regr.', y=1.1)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')



#################### SKLEARN LOGISTIC REGRESSION


#pipe = make_pipeline(StandardScaler(), LogisticRegression())
#pipe.fit(X_validation, y_validation)  # apply scaling on training data
#
## Predict the validation data
#y_prediction_validation = pd.Series(pipe.predict(X_validation))
#
#
#
#
#cnf_matrix = metrics.confusion_matrix(y_validation, y_prediction_validation)
#labels = [0, 1]
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(labels))
#plt.xticks(tick_marks, labels)
#plt.yticks(tick_marks, labels)
## create heatmap
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
#ax.xaxis.set_label_position("top")
#plt.title('Confusion matrix: Log. Regr.', y=1.1)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#
#print("TotalCosts:", totalCosts(y_validation, y_prediction_validation))



##################### Create a logistic regression classifier with cost-sensitive loss function

# defining a custom loss function
def cost_sensitive_loss(cost_TP, cost_FP, cost_FN):
    def loss_function(y_true, y_pred):
        cost = y_true * K.log(y_pred) * cost_FN + y_true * K.log(1 - y_pred) * cost_TP + (1 - y_true) * K.log(1 - y_pred) * cost_FP 
        return  -K.mean(cost, axis=-1)
    return loss_function

# Defining cost for classification mistakes
cost_FP=0.025#25
cost_TP=-0.005#-5
cost_FN=0.005#5

# Build a logistic regression model with Keras
model = Sequential()
model.add(Dense(1, input_shape=(9,), activation='sigmoid', kernel_regularizer=l2(0.)))
model.compile(optimizer='sgd', loss=cost_sensitive_loss(cost_TP,cost_FP, 
            cost_FN))

# Create a pipeline to scale the data and make a prediction on the VALIDATION SET
pipe = make_pipeline(StandardScaler(), model)
pipe.fit(train, train_labels, sequential__nb_epoch=100)
y_prediction_validation = pd.Series(pipe.predict(X_validation).ravel())
y_prediction_validation = (y_prediction_validation>0.5)
y_validation = y_validation.reset_index(drop=True)
print("Custom loss function LR - validation set")
print("Accuracy:", metrics.accuracy_score(y_validation, y_prediction_validation))
print("Precision:", metrics.precision_score(y_validation, y_prediction_validation))
print("Recall:", metrics.recall_score(y_validation, y_prediction_validation))
print("f1-score:", metrics.f1_score(y_validation, y_prediction_validation))
print("TotalCosts:", totalCosts(y_validation, y_prediction_validation))

# create a confusion matrix
cnf_matrix = metrics.confusion_matrix(y_validation, y_prediction_validation)
labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix: Log. Regr.', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')



# Create a Pipeline to scale the data and make a prediction on the TEST SET
y_prediction_test = pd.Series(pipe.predict(test).ravel())
y_prediction_test = (y_prediction_test>0.5)
y_validation = y_validation.reset_index(drop=True)
print("Custom loss function LR - test set")
print("Accuracy:", metrics.accuracy_score(test_labels, y_prediction_test))
print("Precision:", metrics.precision_score(test_labels, y_prediction_test))
print("Recall:", metrics.recall_score(test_labels, y_prediction_test))
print("f1-score:", metrics.f1_score(test_labels, y_prediction_test))
print("TotalCosts:", totalCosts(test_labels, y_prediction_test))

# create a confusion matrix
cnf_matrix = metrics.confusion_matrix(test_labels, y_prediction_test)
labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix: Log. Regr.', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')






print("--------------------------------------------")


####################### Run the logistic regression with balanced class weights
pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced'))
pipe.fit(train, train_labels)  # apply scaling on training data
# Predict the validation data
y_prediction_validation = pd.Series(pipe.predict(train)) # apply the scaling on testing data and get prediction
y_validation = y_validation.reset_index(drop=True)
print("LR-balanced class weights - validation set")
print("Accuracy:", metrics.accuracy_score(train_labels, y_prediction_validation))
print("Precision:", metrics.precision_score(train_labels, y_prediction_validation))
print("Recall:", metrics.recall_score(train_labels, y_prediction_validation))
print("f1-score:", metrics.f1_score(train_labels, y_prediction_validation))
print("TotalCosts:", totalCosts(train_labels, y_prediction_validation))

# Die Accuracy, Precision, Recall und f1-Score für beide Klassen anzeigen lassen
# print("Accuracy:", metrics.accuracy_score(y_validation, y_prediction_validation))
# print("Precision:", metrics.precision_score(y_validation, y_prediction_validation,average=None))
# print("Recall:", metrics.recall_score(y_validation, y_prediction_validation,average=None))
# print("f1-score:", metrics.f1_score(y_validation, y_prediction_validation,average=None))

#### Confusion matrix for prediction on validation set
cnf_matrix = metrics.confusion_matrix(train_labels, y_prediction_validation)
labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix: Log. Regr. w class_weight=balanced', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')

TN = cnf_matrix[0][0]
FN = cnf_matrix[1][0]
TP = cnf_matrix[1][1]
FP = cnf_matrix[0][1]
print("TN:",TN,"FN:",FN,"TP:",TP,"FP:",FP)

# Predict the test data
y_prediction_test = pd.Series(pipe.predict(test)) # apply the scaling on testing data and get prediction
print("LR-balanced class weights - test set")
print("Accuracy:", metrics.accuracy_score(test_labels, y_prediction_test))
print("Precision:", metrics.precision_score(test_labels, y_prediction_test))
print("Recall:", metrics.recall_score(test_labels, y_prediction_test))
print("f1-score:", metrics.f1_score(test_labels, y_prediction_test))
print("TotalCosts:", totalCosts(test_labels, y_prediction_test))





print("--------------------------------------------")




####################### Run the logistic regression with different class weights with scoring function precision
#Setting the range for class weights
weights = np.linspace(0.0,0.99,200)
#Creating a dictionary grid for grid search
param_grid = {'lr__class_weight': [{0:x, 1:1.0-x} for x in weights]}
#pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe = Pipeline([("sc",StandardScaler()),("lr", LogisticRegression())])
#Fitting grid search to the train data with 5 fold stratified fold, focus on improving precision
gridsearch = GridSearchCV(estimator= pipe, 
                          param_grid= param_grid,
                          cv=StratifiedKFold(), 
                          n_jobs=-1, 
                          scoring='precision', 
                          verbose=1).fit(train, train_labels)

#### Ploting the score for different values of weights
sns.set_style('whitegrid')
plt.figure(figsize=(12,8))
weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weigh_data['weight'], weigh_data['score'])
plt.xlabel('Weight for class 2 (y=1)')
plt.ylabel('Precision')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)
#### 
# Get the best weights
weight1,weight2 = list(gridsearch.best_params_.values())[0][0], list(gridsearch.best_params_.values())[0][1]
# Make the prediction again with best weights
pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0: weight1, 1: weight2}))
pipe.fit(train, train_labels)  # apply scaling on training data
# Predict the validation data
y_prediction_training = pd.Series(pipe.predict(train)) # apply the scaling on testing data and get prediction
y_validation = y_validation.reset_index(drop=True)
print("LR-grid search on weights, precision as scoring function - training set")
print("Accuracy:", metrics.accuracy_score(train_labels, y_prediction_training))
print("Precision:", metrics.precision_score(train_labels, y_prediction_training))
print("Recall:", metrics.recall_score(train_labels, y_prediction_training))
print("f1-score:", metrics.f1_score(train_labels, y_prediction_training))
print("TotalCosts:", totalCosts(train_labels, y_prediction_training))

# Die Accuracy, Precision, Recall und f1-Score für beide Klassen anzeigen lassen
# print("Accuracy:", metrics.accuracy_score(y_validation, y_prediction_validation))
# print("Precision:", metrics.precision_score(y_validation, y_prediction_validation,average=None))
# print("Recall:", metrics.recall_score(y_validation, y_prediction_validation,average=None))
# print("f1-score:", metrics.f1_score(y_validation, y_prediction_validation,average=None))

#### Confusion matrix for prediction on validation set
cnf_matrix = metrics.confusion_matrix(train_labels, y_prediction_training)
labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix: Log. Regr. w best weights', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Predict the test data
y_prediction_test = pd.Series(pipe.predict(test)) # apply the scaling on testing data and get prediction
print("LR-grid search on weights, precision as scoring function - test set")
print("Accuracy:", metrics.accuracy_score(test_labels, y_prediction_test))
print("Precision:", metrics.precision_score(test_labels, y_prediction_test))
print("Recall:", metrics.recall_score(test_labels, y_prediction_test))
print("f1-score:", metrics.f1_score(test_labels, y_prediction_test))
print("TotalCosts:", totalCosts(test_labels, y_prediction_test))


print("--------------------------------------------")


########################################## Run the logistic regression with different class weights with own scoring function


#### Run the gridsearch
weights = np.linspace(0.0,0.99,200)

#Creating a dictionary grid for grid search
param_grid = {'lr__class_weight': [{0:x, 1:1.0-x} for x in weights]}

#pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe = Pipeline([("sc",StandardScaler()),("lr", LogisticRegression())])

#Fitting grid search to the train data with 5 stratified fold, focus on improving own cost function
gridsearch = GridSearchCV(estimator= pipe, 
                          param_grid= param_grid,
                          cv=StratifiedKFold(), 
                          n_jobs=-1, 
                          scoring=totalCosts_scorer, 
                          verbose=1).fit(train, train_labels)

#### Ploting the score for different values of weight
sns.set_style('whitegrid')
plt.figure(figsize=(12,8))
weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weigh_data['weight'], weigh_data['score'])
plt.xlabel('Weight for class 2 (y=1)')
plt.ylabel('TotalCosts')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)
#### 

# Get the best weights
weight1,weight2 = list(gridsearch.best_params_.values())[0][0], list(gridsearch.best_params_.values())[0][1]
# Make the prediction again with best weights
pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0: weight1, 1: weight2}))
pipe.fit(train, train_labels)  # apply scaling on training data
# Make the prediction on the validation set
y_prediction_train = pd.Series(pipe.predict(train)) # apply the scaling on testing data and get prediction
y_validation = y_validation.reset_index(drop=True)
print("LR-grid search on weights, own scoring function - validation set")
print("Accuracy:", metrics.accuracy_score(train_labels, y_prediction_train))
print("Precision:", metrics.precision_score(train_labels, y_prediction_train))
print("Recall:", metrics.recall_score(train_labels, y_prediction_train))
print("f1-score:", metrics.f1_score(train_labels, y_prediction_train))
print("TotalCosts:", totalCosts(train_labels, y_prediction_train))

# Die Accuracy, Precision, Recall und f1-Score für beide Klassen anzeigen lassen
# print("Accuracy:", metrics.accuracy_score(y_validation, y_prediction_validation))
# print("Precision:", metrics.precision_score(y_validation, y_prediction_validation,average=None))
# print("Recall:", metrics.recall_score(y_validation, y_prediction_validation,average=None))
# print("f1-score:", metrics.f1_score(y_validation, y_prediction_validation,average=None))

#### Confusion matrix for prediction on validation set
cnf_matrix = metrics.confusion_matrix(train_labels, y_prediction_train)
labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix: Log. Regr. w best weights', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# Predict the test data
y_prediction_test = pd.Series(pipe.predict(test)) # apply the scaling on testing data and get prediction
print("LR-grid search on weights, own scoring function - test set")
print("Accuracy:", metrics.accuracy_score(test_labels, y_prediction_test))
print("Precision:", metrics.precision_score(test_labels, y_prediction_test))
print("Recall:", metrics.recall_score(test_labels, y_prediction_test))
print("f1-score:", metrics.f1_score(test_labels, y_prediction_test))
print("TotalCosts:", totalCosts(test_labels, y_prediction_test))



print("--------------------------------------------")



################### Tuning classification threshold of last model
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')
# predict probabilities
yhat = pipe.predict_proba(X_validation)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = arange(0, 1, 0.001)
# evaluate each threshold
scores = [totalCosts(y_validation,to_labels(probs, t) ) for t in thresholds]
# get best threshold
ix = argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

#### Confusion matrix for prediction on validation set
y_pred_best_threshold = (pipe.predict_proba(train)[:,1]>=thresholds[ix]).astype(int)


cnf_matrix = metrics.confusion_matrix(train_labels, y_pred_best_threshold)
labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix: Log. Regr. w best weights and threshold', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# Predict the train data
y_pred_best_threshold_train = (pipe.predict_proba(train)[:,1]>=thresholds[ix]).astype(int) #prediction results mit verändertem threshold # apply the scaling on testing data and get prediction
y_validation = y_validation.reset_index(drop=True)
print("LR-grid search on weights, own scoring function,optimized threshold - train set")
print("Accuracy:", metrics.accuracy_score(train_labels, y_pred_best_threshold_train ))
print("Precision:", metrics.precision_score(train_labels, y_pred_best_threshold_train ))
print("Recall:", metrics.recall_score(train_labels, y_pred_best_threshold_train ))
print("f1-score:", metrics.f1_score(train_labels, y_pred_best_threshold_train ))
print("TotalCosts:", totalCosts(train_labels, y_pred_best_threshold_train ))

# Predict the test data
y_pred_best_threshold_test = (pipe.predict_proba(test)[:,1]>=thresholds[ix]).astype(int) # apply the scaling on testing data and get prediction
print("LR-grid search on weights, own scoring function,optimized threshold - test set")
print("Accuracy:", metrics.accuracy_score(test_labels, y_prediction_test))
print("Precision:", metrics.precision_score(test_labels, y_prediction_test))
print("Recall:", metrics.recall_score(test_labels, y_prediction_test))
print("f1-score:", metrics.f1_score(test_labels, y_prediction_test))
print("TotalCosts:", totalCosts(test_labels, y_prediction_test))


############################################## Prediction auf test set

# y_pred_best_threshold_TEST = (pipe.predict_proba(test)[:,1]>=thresholds[ix]).astype(int) #prediction results mit verändertem threshold
# y_so = pipe.predict(test)#prediction results ohne veränderten threshold
# #totalCosts(test_labels,y_so) #Kosten berechnen

# cnf_matrix = metrics.confusion_matrix(test_labels, y_pred_best_threshold_TEST)
# labels = [0, 1]
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(labels))
# plt.xticks(tick_marks, labels)
# plt.yticks(tick_marks, labels)
# # create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlOrRd", fmt='g')
# ax.xaxis.set_label_position("top")
# plt.title('Confusion matrix: Log. Regr. w best weights and threshold', y=1.1)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')






























