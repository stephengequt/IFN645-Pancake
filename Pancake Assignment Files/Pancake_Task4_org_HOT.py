# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 11:44:23 2018

@author: Soam wei jie
"""

import os

print(os.getcwd())
def data_prep():
    
    # read the organics dataset 
    org1 = pd.read_csv('Organic_Clean.csv')
    
    #drop variables
    org1.drop(['ORGANICS','AGEGRP1'],axis = 1, inplace = True)
  
    #one-hot encoding
    org1 = pd.get_dummies(org1)

    print(org1.info())
    
    return org1
       


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
#from dm_tools import data_prep
from sklearn.preprocessing import StandardScaler

rs = 10

# train test split
org1 = data_prep()
y = org1['ORGYN']
X = org1.drop(['ORGYN'], axis=1)
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

model = MLPClassifier(max_iter=200 , random_state=rs)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


print(model)

########################################################################################
### Grid Search Neural network #######

print(X_train.shape)

params = {'hidden_layer_sizes': [(3,)],
          'alpha': [0.01],
          'solver': ['adam'],
          'activation': ['identity']}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs, max_iter = 300), cv=10, n_jobs=-1)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

print(cv.best_estimator_.n_iter_)

print(cv.best_params_)

################## features importance ######################

print("\n===================visualize feature importance================")
coef = cv.best_estimator_.coefs_[0]

feature_importance = list()
for item in coef:
#    print(item[])
    importance = np.abs(item[0] + item[1])
    feature_importance.append(importance)
    
import matplotlib.pyplot as plt
feature_importance = pd.Series(feature_importance, index =  X.columns)
feature_importance = feature_importance.sort_values()
plt.figure(figsize = (6,12))
feature_importance.plot.barh()

##################### New parameters ############################

print(X_train.shape)

params = {'hidden_layer_sizes': [(x,) for x in range(1, 9, 1)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

print(cv.best_params_)

params = {'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,)]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

print(cv.best_estimator_.n_iter_)

print(cv.best_params_)

########### Test with added 'Alpha parameter ###################

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
X_log = df_log.drop(['ORGYN'], axis=1)
X_mat_log = X_log.as_matrix()
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, 
                                                                    random_state=rs)

# standardise them again
scaler_log = StandardScaler()
X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
X_test_log = scaler_log.transform(X_test_log)

############ RETRAIN after LOG TRANSFORMATION #############

params = {'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_log, y_train_log)

print("Train accuracy:", cv.score(X_train_log, y_train_log))
print("Test accuracy:", cv.score(X_test_log, y_test_log))

y_pred = cv.predict(X_test_log)
print(classification_report(y_test_log, y_pred))

print(cv.best_params_)

# Accuracy of test actually decreases!!!!
#########################################################
############   LOG Transformation #################################
import numpy as np

# list columns to be transformed
columns_to_transform = ['LTIME','AFFL']

# copy the dataframe
df_log = org1.copy()

# transform the columns with np.log
for col in columns_to_transform:
    df_log[col] = df_log[col].apply(lambda x: x+1)
    df_log[col] = df_log[col].apply(np.log)
    
# create X, y and train test data partitions
y_log = df_log['ORGYN']
X_log = df_log.drop(['ORGYN'], axis=1)
X_mat_log = X_log.as_matrix()
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, 
                                                                    random_state=rs)

# standardise them again
scaler_log = StandardScaler()
X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
X_test_log = scaler_log.transform(X_test_log)
###########################################################
######## Recursive elimination ##########
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

rfe = RFECV(estimator = LogisticRegression(random_state=rs, C = 1, multi_class = 'ovr', penalty = 'l2', solver = 'newton-cg'), cv=10)
rfe.fit(X_train_log, y_train_log)

print(rfe.n_features_)
# grab feature importances from the model and feature name from the original X
idxs_selected = rfe.get_support(indices=True)
feature_names = X.columns
selected_features = feature_names[idxs_selected]
print(selected_features)
features_score = rfe.grid_scores_[idxs_selected]

# sort them out in descending order
indices = np.argsort(np.absolute(features_score))
indices = np.flip(indices, axis=0)

for i in indices:

    print(selected_features[i], ':', features_score[i])
# transform log 
X_train_rfe = rfe.transform(X_train_log)
X_test_rfe = rfe.transform(X_test_log)

# step = int((X_train_rfe.shape[1] + 5)/5);
params = {'hidden_layer_sizes': [(x,) for x in range(4, 16, 4)], 'alpha': [0.01,0.001,0.0001,0.00001], 'solver': ['adam', 'lbfgs'], 'activation': ['identity','relu']}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_rfe, y_train_log)
log_reg_model = cv.best_estimator_

print("Train accuracy:", cv.score(X_train_rfe, y_train_log))
print("Test accuracy:", cv.score(X_test_rfe, y_test_log))

y_pred = cv.predict(X_test_rfe)
print(classification_report(y_test_log, y_pred))
print(cv.best_estimator_.n_iter_)

print(cv.best_params_)

##################### Selection using Decision Tree ####################
from sklearn.tree import DecisionTreeClassifier

params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(3, 8),
          'min_samples_leaf': range(20, 61, 10)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train_log, y_train_log)

from dm_tools import analyse_feature_importance

analyse_feature_importance(cv.best_estimator_, X_log.columns)

from sklearn.feature_selection import SelectFromModel

selectmodel = SelectFromModel(cv.best_estimator_, prefit=True)
X_train_sel_model = selectmodel.transform(X_train)
X_test_sel_model = selectmodel.transform(X_test)

print(X_train_sel_model.shape)


params = {'hidden_layer_sizes': [(1,), (2,), (3,), (4,)], 'alpha': [0.01,0.001,0.0001,0.00001], 'solver': ['adam','lbfgs'], 'activation': ['relu']}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_sel_model, y_train)

print("Train accuracy:", cv.score(X_train_sel_model, y_train))
print("Test accuracy:", cv.score(X_test_sel_model, y_test))

y_pred = cv.predict(X_test_sel_model)
print(classification_report(y_test, y_pred))
print(cv.best_estimator_.n_iter_)

print(cv.best_params_)


################# Model Comparison ##########################
### Logistic Regression is chosen over decision tree for model comparison because the performance of regression is better than DT. 

params_dt = {'criterion': ['gini'],
          'max_depth': range(5, 6),
          'min_samples_leaf': range(30, 61, 5)}

cv = GridSearchCV(param_grid=params_dt, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train, y_train)

dt_model = cv.best_estimator_
print(dt_model)

print(cv.best_params_)

nn_model = MLPClassifier(random_state=rs, activation = 'identity' , alpha = 0.01 , hidden_layer_sizes = (3,) , solver = 'adam' )
nn_model.fit(X_train, y_train)
print(nn_model.score(X_train,y_train))
print(nn_model.score(X_test,y_test))

y_pred_dt = dt_model.predict(X_test)
y_pred_log_reg = log_reg_model.predict(X_test_rfe)
y_pred_nn = nn_model.predict(X_test)

print("Accuracy score on test for DT:", accuracy_score(y_test, y_pred_dt))
print("Accuracy score on test for logistic regression:", accuracy_score(y_test, y_pred_log_reg))
print("Accuracy score on test for NN:", accuracy_score(y_test, y_pred_nn))

####################### roc  ##################
from sklearn.metrics import roc_auc_score

y_pred_proba_dt = dt_model.predict_proba(X_test)
y_pred_proba_log_reg = log_reg_model.predict_proba(X_test_rfe)
y_pred_proba_nn = nn_model.predict_proba(X_test)

roc_index_dt = roc_auc_score(y_test, y_pred_proba_dt[:, 1])
roc_index_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg[:, 1])
roc_index_nn = roc_auc_score(y_test, y_pred_proba_nn[:, 1])

print("ROC index on test for DT:", roc_index_dt)
print("ROC index on test for logistic regression:", roc_index_log_reg)
print("ROC index on test for NN:", roc_index_nn)

######### plotting ROC ############

from sklearn.metrics import roc_curve

fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_proba_dt[:,1])
fpr_log_reg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, y_pred_proba_log_reg[:,1])
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_pred_proba_nn[:,1])

import matplotlib.pyplot as plt

plt.plot(fpr_dt, tpr_dt, label='ROC Curve for DT {:.3f}'.format(roc_index_dt), color='red', lw=0.5)
plt.plot(fpr_log_reg, tpr_log_reg, label='ROC Curve for Log reg {:.3f}'.format(roc_index_log_reg), color='green', lw=0.5)
plt.plot(fpr_nn, tpr_nn, label='ROC Curve for NN {:.3f}'.format(roc_index_nn), color='darkorange', lw=0.5)

# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


###################### Ensemble model #############################


from sklearn.ensemble import VotingClassifier

# initialise the classifier with 3 different estimators
voting = VotingClassifier(estimators=[('dt', dt_model), ('lr', log_reg_model), ('nn', nn_model)], voting='soft')

# fit the voting classifier to training data
voting.fit(X_train, y_train)
print(voting.score(X_train,y_train))
print(voting.score(X_test,y_test))

# evaluate train and test accuracy
print("Ensemble train accuracy:", voting.score(X_train, y_train))
print("Ensemble test accuracy:", voting.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# evaluate ROC auc score
y_pred_proba_ensemble = voting.predict_proba(X_test)
roc_index_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble[:, 1])
print("ROC score of voting classifier:", roc_index_ensemble)


#################### MODEL COMPARISON  ######################################

y_pred_dt = dt_model.predict(X_test)
y_pred_log_reg = log_reg_model.predict(X_test_rfe)
y_pred_nn = nn_model.predict(X_test)
y_pred_em = voting.predict(X_test)

print("Accuracy score on test for DT:", accuracy_score(y_test, y_pred_dt))
print("Accuracy score on test for logistic regression:", accuracy_score(y_test, y_pred_log_reg))
print("Accuracy score on test for NN:", accuracy_score(y_test, y_pred_nn))
print("Accuracy score on test for EM:", accuracy_score(y_test, y_pred_em))


from sklearn.metrics import roc_auc_score

y_pred_proba_dt = dt_model.predict_proba(X_test)
y_pred_proba_log_reg = log_reg_model.predict_proba(X_test_rfe)
y_pred_proba_nn = nn_model.predict_proba(X_test)
y_pred_proba_em = voting.predict(X_test)

roc_index_dt = roc_auc_score(y_test, y_pred_proba_dt[:, 1])
roc_index_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg[:, 1])
roc_index_nn = roc_auc_score(y_test, y_pred_proba_nn[:, 1])
roc_index_em = roc_auc_score(y_test, y_pred_proba_ensemble[:, 1])

print("ROC index on test for DT:", roc_index_dt)
print("ROC index on test for logistic regression:", roc_index_log_reg)
print("ROC index on test for NN:", roc_index_nn)
print("ROC index on test for EM:", roc_index_em)


########################################################################

from sklearn.metrics import roc_curve

fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_proba_dt[:,1])
fpr_log_reg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, y_pred_proba_log_reg[:,1])
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_pred_proba_nn[:,1])
fpr_em, tpr_em, thresholds_em = roc_curve(y_test, y_pred_proba_ensemble[:,1])

import matplotlib.pyplot as plt

plt.plot(fpr_dt, tpr_dt, label='ROC Curve for DT {:.3f}'.format(roc_index_dt), color='red', lw=0.5)
plt.plot(fpr_log_reg, tpr_log_reg, label='ROC Curve for Log reg {:.3f}'.format(roc_index_log_reg), color='green', lw=0.5)
plt.plot(fpr_nn, tpr_nn, label='ROC Curve for NN {:.3f}'.format(roc_index_nn), color='darkorange', lw=0.5)
plt.plot(fpr_em, tpr_em, label='ROC Curve for EM {:.3f}'.format(roc_index_em), color='blue', lw=0.5)

# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


################### Classification report for ensemble ##########################################################

