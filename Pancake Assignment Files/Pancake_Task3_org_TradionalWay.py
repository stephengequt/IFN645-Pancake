# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:30:33 2018

@author: Soam wei jie
"""
import os

os.getcwd()
def data_prep():
    
    # read the organics dataset 
    org1 = pd.read_csv('Organic_Clean.csv')
    
    # drop variables
    #org1.drop(['AGEGRP1'],axis = 1, inplace = True)
  
    # one-hot encoding
    #org1 = pd.get_dummies(org1)

    print(org1.info())
    
    return org1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


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


# set the random seed - consistent
rs = 10

# train test split
y = org1['ORGYN']
X = org1.drop(['ORGYN','ORGANICS'], axis=1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)


from sklearn.preprocessing import StandardScaler

# initialise a standard scaler object
scaler = StandardScaler()

# visualise min, max, mean and standard dev of data before scaling
print("Before scaling\n-------------")
for i in range(5):
    col = X_train[:,i]
    print("Variable #{}: min {}, max {}, mean {:.2f} and std dev {:.2f}".
          format(i, min(col), max(col), np.mean(col), np.std(col)))

# learn the mean and std.dev of variables from training data
# then use the learned values to transform training data
X_train = scaler.fit_transform(X_train, y_train)

print("After scaling\n-------------")
for i in range(5):
    col = X_train[:,i]
    print("Variable #{}: min {}, max {}, mean {:.2f} and std dev {:.2f}".
          format(i, min(col), max(col), np.mean(col), np.std(col)))

# use the statistic that you learned from training to transform test data
# NEVER learn from test data, this is supposed to be a set of dataset
# that the model has never seen before
X_test = scaler.transform(X_test)

#####################################################################################

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=rs)

# fit it to training data
model.fit(X_train, y_train)

# training and test accuracy
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# classification report on test data
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print(model.coef_)

feature_names = X.columns
coef = model.coef_[0]

# limit to 20 features, you can comment the following line to print out everything
coef = coef[:20]

for i in range(len(coef)):
    print(feature_names[i], ':', coef[i])
    
# grab feature importances from the model and feature name from the original X
coef = model.coef_[0]
feature_names = X.columns

# sort them out in descending order
indices = np.argsort(np.absolute(coef))
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:20]

for i in indices:
    print(feature_names[i], ':', coef[i])  
    
######################################## grid search CV ###################################
params = {'C': [pow(10, x) for x in range(-6, 4)]}

# use all cores to tune logistic regression with C parameter
cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train, y_train)

# test the best model
print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

# print parameters of the best model
print(cv.best_params_)


#############################################################################################3

import seaborn as sns
import matplotlib.pyplot as plt

def plot_skewed_columns(df):
    # setting up subplots for easier visualisation
    f, axes = plt.subplots(2,4, figsize=(10,10), sharex=False)

    # gift avg plots
    sns.distplot(df['GiftAvg36'].dropna(), hist=False, ax=axes[0,0])
    sns.distplot(df['GiftAvgAll'].dropna(), hist=False, ax=axes[0,1])
    sns.distplot(df['GiftAvgCard36'].dropna(), hist=False, ax=axes[1,0])
    sns.distplot(df['GiftAvgLast'].dropna(), hist=False, ax=axes[1,1])

    # gift cnt plots
    sns.distplot(df['GiftCnt36'].dropna(), hist=False, ax=axes[0,2])
    sns.distplot(df['GiftCntAll'].dropna(), hist=False, ax=axes[0,3])
    sns.distplot(df['GiftCntCard36'].dropna(), hist=False, ax=axes[1,2])
    sns.distplot(df['GiftCntCardAll'].dropna(), hist=False, ax=axes[1,3])

    plt.show()
    
plot_skewed_columns(df)

###### Transform variables #########
import numpy as np

# list columns to be transformed
columns_to_transform = ['GiftAvg36', 'GiftAvgAll', 'GiftAvgCard36', 'GiftAvgLast',
                        'GiftCnt36', 'GiftCntAll', 'GiftCntCard36', 'GiftCntCardAll']

# copy the dataframe
df_log = df.copy()

# transform the columns with np.log
for col in columns_to_transform:
    df_log[col] = df_log[col].apply(lambda x: x+1)
    df_log[col] = df_log[col].apply(np.log)

# plot them again to show the distribution
plot_skewed_columns(df_log)

####### RESCALE and RESAMPLE training and test data #########

# create X, y and train test data partitions
y_log = df_log['TargetB']
X_log = df_log.drop(['TargetB'], axis=1)
X_mat_log = X_log.as_matrix()
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, 
                                                                    random_state=rs)

# standardise them again
scaler_log = StandardScaler()
X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
X_test_log = scaler_log.transform(X_test_log)

##################################################################

# grid search CV
params = {'C': [pow(10, x) for x in range(-6, 4)]}

cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_log, y_train_log)

# test the best model
print("Train accuracy:", cv.score(X_train_log, y_train_log))
print("Test accuracy:", cv.score(X_test_log, y_test_log))

y_pred = cv.predict(X_test_log)
print(classification_report(y_test_log, y_pred))

# print parameters of the best model
print(cv.best_params_)

######################################################################

########## Dimensionality elimination ###############

from sklearn.feature_selection import RFECV

rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
rfe.fit(X_train, y_train) # run the RFECV

# comparing how many variables before and after
print("Original feature set", X_train.shape[1])
print("Number of features after elimination", rfe.n_features_)

X_train_sel = rfe.transform(X_train)
X_test_sel = rfe.transform(X_test)

######## Re-Run the model using grid search ##########
# grid search CV
params = {'C': [pow(10, x) for x in range(-6, 4)]}

cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_sel, y_train)

# test the best model
print("Train accuracy:", cv.score(X_train_sel, y_train))
print("Test accuracy:", cv.score(X_test_sel, y_test))

y_pred = cv.predict(X_test_sel)
print(classification_report(y_test, y_pred))

# print parameters of the best model
print(cv.best_params_)


######################################################################################################3
# running RFE + log transformation
rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
rfe.fit(X_train_log, y_train_log) # run the RFECV on log transformed dataset

# comparing how many variables before and after
print("Original feature set", X_train_log.shape[1])
print("Number of features after elimination", rfe.n_features_)

# select features from log transformed dataset
X_train_sel_log = rfe.transform(X_train_log)
X_test_sel_log = rfe.transform(X_test_log)

# init grid search CV on transformed dataset
params = {'C': [pow(10, x) for x in range(-6, 4)]}
cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_sel_log, y_train_log)

# test the best model
print("Train accuracy:", cv.score(X_train_sel_log, y_train_log))
print("Test accuracy:", cv.score(X_test_sel_log, y_test_log))

y_pred_log = cv.predict(X_test_sel_log)
print(classification_report(y_test_log, y_pred_log))

# print parameters of the best model
print(cv.best_params_)