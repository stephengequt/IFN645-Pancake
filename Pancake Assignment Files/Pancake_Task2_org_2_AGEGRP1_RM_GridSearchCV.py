#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:06:35 2018

@author: aaron
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:45:03 2018

@author: aaron
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# data_prep
    
# read the organics dataset 
org_2 = pd.read_csv('Organic_Clean.csv')

# drop GRPAGE1 & GRPAGE2 variables and create seperate datasets
org_2.drop(['AGEGRP1'], axis=1, inplace=True)
    
# one-hot encoding all files
org_2 = pd.get_dummies(org_2)

print(org_2.info())    

# Testing data with org_1 dataset
# target/input split
y = org_2['ORGYN']
X = org_2.drop(['ORGYN', 'ORGANICS'], axis=1)

# setting random state
rs = 10

X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

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
plt.plot(range(2, 21), train_score, 'b', range(2,21), test_score, 'r')
plt.xlabel('max_depth\nBlue = Training Acc. Red = Test Acc.')
plt.ylabel('accuracy')
plt.show()


# find optimal hyperparamters with GridSearchCV
from sklearn.model_selection import GridSearchCV

#grid search CV
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2,6),
          'min_samples_leaf': range(20,50,10)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

#test the best model
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

#print parameters of the best model
print(cv.best_params_)

#grid search CV #2
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(4,6),
          'min_samples_leaf': range(20,40)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

#test the best model
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

#print parameters of the best model
print(cv.best_params_)

# rerun decision tree training and testing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# simple decision tree training
model = DecisionTreeClassifier(random_state=rs, criterion='gini', max_depth=5, min_samples_leaf=29)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Check feature importance
import numpy as np

# grab feature importances from the model and feature name from the original X
importances = model.feature_importances_
feature_names = X.columns

#sort them out in descending order
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

#limit to 10 features
indices = indices[:10]

for i in indices:
    print(feature_names[i], ":", importances[i])

# Visualising
import pydot
from io import StringIO
from sklearn.tree import export_graphviz

dotfile = StringIO()
export_graphviz(model, out_file = dotfile, feature_names = X.columns)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph[0].write_png("Pancake_Decision_Tree_GridSearchCV.png")

# View nodes
model.tree_.children_left #array of left children
model.tree_.children_right #array of right children
model.tree_.feature #array of nodes splitting feature
model.tree_.threshold #array of nodes splitting points
model.tree_.value #array of nodes values

from inspect import getmembers
print(getmembers(model.tree_))