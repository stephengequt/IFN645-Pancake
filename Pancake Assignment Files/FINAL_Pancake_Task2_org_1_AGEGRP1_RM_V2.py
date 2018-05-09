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
org_1 = pd.read_csv('Organic_Clean.csv')

# drop GRPAGE1 & GRPAGE2 variables and create seperate datasets
org_1.drop(['AGEGRP1'], axis=1, inplace=True)
    
# one-hot encoding all files
org_1 = pd.get_dummies(org_1)

print(org_1.info())    

# Testing data with org_1 dataset
# target/input split
y = org_1['ORGYN']
X = org_1.drop(['ORGYN', 'ORGANICS'], axis=1)

# setting random state
rs = 10

X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

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
import numpy as np

# grab feature importances from the model and feature name from the original X
importances = model.feature_importances_
feature_names = X.columns

#sort them out in descending order
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

#number of nodes
print(model.tree_.node_count)

#number of leaves 
leave_id = model.apply(X_train)
leave_id = np.unique(leave_id)
print(len(leave_id))

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
graph[0].write_png("Pancake_Decision_Tree_1.png")

#variable is used for the first split and competing splits for this first split
feature_names[model.tree_.feature]

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# hyperparameters and model performance
# max_leaf_nodes
test_score = []
train_score = []

# check model performance for max depth from 2-200
for max_leaf_nodes in range(2, 200):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=rs)
    model.fit(X_train, y_train)
    
    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))

    
import matplotlib.pyplot as plt

# plot max depth hyperparameter values vs training and test accuracy score
plt.plot(range(2, 200), train_score, 'b', range(2,200), test_score, 'r')
plt.xlabel('max_leaf_nodes\nBlue = Training Acc. Red = Test Acc.')
plt.ylabel('accuracy')
plt.show()


# hyperparameters and model performance
# min_samples_split
test_score = []
train_score = []

# check model performance for max depth from 2-300
for min_samples_split in range(2, 300):
    model = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=rs)
    model.fit(X_train, y_train)
    
    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))

    
import matplotlib.pyplot as plt

# plot max depth hyperparameter values vs training and test accuracy score
plt.plot(range(2, 300), train_score, 'b', range(2,300), test_score, 'r')
plt.xlabel('min_samples_split\nBlue = Training Acc. Red = Test Acc.')
plt.ylabel('accuracy')
plt.show()


# hyperparameters and model performance
# min_samples_leaf
test_score = []
train_score = []

# check model performance for max depth from 2-
for min_samples_split in range(2, 300):
    model = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=rs)
    model.fit(X_train, y_train)
    
    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))

    
import matplotlib.pyplot as plt

# plot max depth hyperparameter values vs training and test accuracy score
plt.plot(range(2, 300), train_score, 'b', range(2,300), test_score, 'r')
plt.xlabel('min_samples_split\nBlue = Training Acc. Red = Test Acc.')
plt.ylabel('accuracy')
plt.show()
