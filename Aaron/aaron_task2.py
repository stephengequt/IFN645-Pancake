#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:45:03 2018

@author: aaron
"""
from sklearn.model_selection import train_test_split

# preprocessing
org1 = pd.read_csv('organics.csv')

# one-hot encoding
org1 = pd.get_dummies(org1)

# display info
print(org1.info())

# target/input split
y = org1['ORGANICS']
X = org1.drop(['ORGANICS'], axis=1)

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