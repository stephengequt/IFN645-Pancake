#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:45:03 2018

@author: aaron
"""
import os

os.getcwd()
def data_prep():
    
    # read the organics dataset 
    org1 = pd.read_csv('Organic_Clean_V2.csv')
    
    # drop variables
    #org1.drop(['ORGANICS','AGEGRP2'],axis = 1, inplace = True)
  
    #one-hot encoding
    #org1 = pd.get_dummies(org1)

    #print(org1.info())
    
    return org1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
#from dm_tools import data_prep

# call data_prep method


org1 = data_prep()
GENDER = {'F':0, 'M':1,'U':2}
org1['GENDER'] = org1['GENDER'].map(GENDER)

AGEGRP1 = {'<20':1, '20-40':2,'40-60':3,'60-80':4}
org1['AGEGRP1'] = org1['AGEGRP1'].map(AGEGRP1)

AGEGRP2 = {'10-20':1,'20-30':2, '30-40':3,'40-50':4,'50-60':5, '60-70':6, '70-80':7}
org1['AGEGRP2'] = org1['AGEGRP2'].map(AGEGRP2)

NGROUP = {'A':0, 'B':1,'C':2, 'D':3, 'E':4, 'F':5, 'U':6}
org1['NGROUP'] = org1['NGROUP'].map(NGROUP)

REGION = {'Midlands':0, 'North':1,'Scottish':2,'South East':3,'South West':4, 'Unknown':5}
org1['REGION'] = org1['REGION'].map(REGION)

CLASS = {'Tin':0,'Silver':1,'Platinum':2,'Gold':3}
org1['CLASS'] = org1['CLASS'].map(CLASS)

AFFL = {'<5':1,'<10':2, '<15':3,'<20':4,'<25':5, '<30':6, '<35':7}
org1['AFFL'] = org1['AFFL'].map(AFFL)

LTIME = {'<5':1,'<10':2, '<15':3,'<20':4,'<25':5, '<30':6, '<35':7, '<40':8}
org1['LTIME'] = org1['LTIME'].map(LTIME)


# target/input split
y = org1['ORGYN']
X = org1.drop(['ORGYN','ORGANICS','AGEGRP1'], axis=1)

# setting random state
rs = 10

X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.4, stratify=y, random_state=rs)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# simple decision tree training
model = DecisionTreeClassifier(random_state=rs)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Check feature importance

# grab feature importances from the model and feature name from the original X
importances = model.feature_importances_
feature_names = X.columns

#sort them out in descending order
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

#limit to 20 features
indices = indices[:20]

for i in indices:
    print(feature_names[i], ":", importances[i])
    
# Visualising
import pydot
from io import StringIO
from sklearn.tree import export_graphviz

dotfile = StringIO()
export_graphviz(model, out_file = dotfile, feature_names = X.columns)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph.write_png("aarontask2_org_vis.png")

#retrain with a small max_depth limit

model = DecisionTreeClassifier(max_depth=3, random_state=rs)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# grab feature importance from the model and feature name from the original X
importances = model.feature_importances_
feature_names = X.columns

# sort them out in descending order
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:20]

for i in indices:
    print(feature_names[i], ':', importances[i])

# visualize
dotfile = StringIO()
export_graphviz(model, out_file=dotfile, feature_names=X.columns)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph.write_png("week3_dt_viz.png") # saved in the following file


# hyperparameters and model performance
test_score = []
train_score = []

# check model performance for max depth from 2-20
for max_depth in range(2, 21):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=rs)
    model.fit(X_train, y_train)
    
    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))
    
import matplotlib.pyplot as plt

# plot max depth hyperparameter values vs training and test accuracy score
plt.plot(range(2, 21), train_score, 'b', range(6,21), test_score, 'r')
plt.xlabel('max_depth\nBlue = Training Acc. Red = Test Acc.')
plt.ylabel('accuracy')
plt.show()


# find optimal hyperparamters with GridSearchCV
from sklearn.model_selection import GridSearchCV

#grid search CV
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2,7),
          'min_samples_leaf': range(20,600,10)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

#test the best model
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

#print parameters of the best model
print(cv.best_params_)

################# importance parameters #####################
importances = model.feature_importances_
feature_names = X.columns

# sort them out in descending order
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:20]

for i in indices:
    print(feature_names[i], ':', importances[i])

#grid search CV #2
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(1,6),
          'min_samples_leaf': range(35,300)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

#test the best model
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

#print parameters of the best model
print(cv.best_params_)


# inside `dm_tools.py' together with data_prep()
import numpy as np
import pydot
from io import StringIO
from sklearn.tree import export_graphviz

def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i], ':', importances[i])

def visualize_decision_tree(dm_model, feature_names, save_name):
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph[0].write_png(save_name) # saved in the following file