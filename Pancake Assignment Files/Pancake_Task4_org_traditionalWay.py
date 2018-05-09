# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 10:55:05 2018

@author: Soam wei jie
"""
import os

os.getcwd()
def data_prep():
    
    # read the organics dataset 
    org1 = pd.read_csv('Organic_Clean.csv')
    
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
from sklearn.preprocessing import StandardScaler

# preprocessing step
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
# random state
rs = 10

# train test split
y = org1['ORGYN']
X = org1.drop(['ORGYN','ORGANICS','AGEGRP1'], axis=1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)


#########################################################################################

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(random_state=rs)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print(model)

model = MLPClassifier(max_iter=100, random_state=rs)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print(model)

########################################################################################
### Grid Search Neural network #######

print(X_train.shape)

params = {'hidden_layer_sizes': [(x,) for x in range(1, 8, 1)]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

print(cv.best_params_)

##### New parameters ########

params = {'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,)]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

print(cv.best_params_)


######## ADD alpha ###################

params = {'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

print(cv.best_params_)

################### LOG TRANSFORMATION ##########################
import numpy as np

# list columns to be transformed
columns_to_transform = ['AFFL','LTIME']

# copy the dataframe
df_log = org1.copy()

# transform the columns with np.log
for col in columns_to_transform:
    df_log[col] = df_log[col].apply(lambda x: x+1)
    df_log[col] = df_log[col].apply(np.log)
    
# create X, y and train test data partitions
y_log = df_log['ORGYN']
X_log = df_log.drop(['ORGYN','ORGANICS','AGEGRP1'], axis=1)
X_mat_log = X_log.as_matrix()
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, 
                                                                    random_state=rs)

# standardise them again
scaler_log = StandardScaler()
X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
X_test_log = scaler_log.transform(X_test_log)


##### Test again ########
### result actually became worst #####

params = {'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_log, y_train_log)

print("Train accuracy:", cv.score(X_train_log, y_train_log))
print("Test accuracy:", cv.score(X_test_log, y_test_log))

y_pred = cv.predict(X_test_log)
print(classification_report(y_test_log, y_pred))

print(cv.best_params_)


######### RECURSIVE FEATURE ELIMINATION ############
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
rfe.fit(X_train_log, y_train_log)

print(rfe.n_features_)

# transform log 
X_train_rfe = rfe.transform(X_train_log)
X_test_rfe = rfe.transform(X_test_log)

# step = int((X_train_rfe.shape[1] + 5)/5);
params = {'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_rfe, y_train_log)

print("Train accuracy:", cv.score(X_train_rfe, y_train_log))
print("Test accuracy:", cv.score(X_test_rfe, y_test_log))

y_pred = cv.predict(X_test_rfe)
print(classification_report(y_test_log, y_pred))

print(cv.best_params_)