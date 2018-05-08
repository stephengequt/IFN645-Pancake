import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def data_prep():
    # read the organics dataset
    org1 = pd.read_csv('Stephen/dataset/Organic_Clean.csv')

    # drop the unused target variable that is ORGANICS
    org1.drop(['ORGANICS', 'AGEGRP1'], axis=1, inplace=True)

    # one-hot encoding
    org1 = pd.get_dummies(org1)

    print(org1.info())

    return org1


# call data_prep method
org1 = data_prep()

#Transformation
print('\n----------------------Before Transformation------------------------\n')
# setting up subplots for easier visualisation
f, axes = plt.subplots(1, 2, figsize=(5, 5), sharex=False)

sns.distplot(org1['AFFL'].dropna(), hist=False, ax=axes[0])
sns.distplot(org1['LTIME'].dropna(), hist=False, ax=axes[1])

plt.show()

# list columns to be transformed
columns_to_transform = ['AFFL', 'LTIME']

# copy the dataframe
org1_log = org1.copy()

# transform the columns with np.log
for col in columns_to_transform:
    org1_log[col] = org1_log[col].apply(lambda x: x + 1)
    org1_log[col] = org1_log[col].apply(np.log)

# plot them again to show the distribution
print('\n----------------------After Transformation------------------------\n')

# setting up subplots for easier visualisation
f, axes = plt.subplots(1, 2, figsize=(5, 5), sharex=False)

sns.distplot(org1_log['AFFL'].dropna(), hist=False, ax=axes[0])
sns.distplot(org1_log['LTIME'].dropna(), hist=False, ax=axes[1])

plt.show()


# create X, y and train test data partitions
rs = 10
y_log = org1_log['ORGYN']
X_log = org1_log.drop(['ORGYN'], axis=1)
X_mat_log = X_log.as_matrix()
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log,
                                                                    random_state=rs)

# standardise
print("\n------------------Before scaling------------------\n")
for i in range(8):
    col = X_train_log[:, i]
    print("Variable #{}: min {}, max {}, mean {:.2f} and std dev {:.2f}".
          format(i, min(col), max(col), np.mean(col), np.std(col)))

scaler_log = StandardScaler()
X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
X_test_log = scaler_log.transform(X_test_log)

print("\n-------------------After scaling-------------------\n")
for i in range(8):
    col = X_train_log[:, i]
    print("Variable #{}: min {}, max {}, mean {:.2f} and std dev {:.2f}".
          format(i, min(col), max(col), np.mean(col), np.std(col)))



print("\n-------------------Default LogisticRegression Model-------------------\n")

model = LogisticRegression(random_state=rs)

# fit it to training data
model.fit(X_train_log, y_train_log)

# training and test accuracy
print("Train accuracy:", model.score(X_train_log, y_train_log))
print("Test accuracy:", model.score(X_test_log, y_test_log))

# classification report on test data
y_pred = model.predict(X_test_log)
print(classification_report(y_test_log, y_pred))




print("\n--------------------- Optimal LogisticRegression Model-------------------\n")

# grid search CV
params = {'C': [pow(10, x) for x in range(-6, 4)],
          'penalty': ['l2'],
          'solver': ['newton-cg', 'lbfgs', 'sag']
          }

# grid search CV
params2 = {'C': [pow(10, x) for x in range(-6, 4)],
          'penalty': ['l1'],
          'solver': ['liblinear']
           }

# use all cores to tune logistic regression with C parameter
cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_log, y_train_log)


# test the best model
print("Train accuracy:", cv.score(X_train_log, y_train_log))
print("Test accuracy:", cv.score(X_test_log, y_test_log))

y_pred = cv.predict(X_test_log)
print(classification_report(y_test_log, y_pred))

# print parameters of the best model
print(cv.best_params_)


print("\n--------------------------Feature Importance-------------------------\n")
feature_names = X_log.columns
coef = model.coef_[0]

# limit to 20 features, you can comment the following line to print out everything
# coef = coef[:20]

for i in range(len(coef)):
    print(feature_names[i], ':', coef[i])

feature_importance = pd.Series(np.abs(coef), index = feature_names)
feature_importance=feature_importance.sort_values(ascending=False)

from seaborn import set_style
set_style('dark')
plt.figure(figsize=[10,7])
(feature_importance.plot.bar())
plt.show()


print("\n--------------------------Recursive feature elimination-------------------------\n")

from sklearn.feature_selection import RFECV

rfe = RFECV(estimator=LogisticRegression(random_state=rs), cv=10)
rfe.fit(X_train_log, y_train_log)  # run the RFECV

# comparing how many variables before and after
print("Original feature set", X_train_log.shape[1])
print("Number of features after elimination", rfe.n_features_)

X_train_sel_log = rfe.transform(X_train_log)
X_test_sel_log = rfe.transform(X_test_log)
print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), feature_names)))

# init grid search CV on transformed dataset
cv.fit(X_train_sel_log, y_train_log)

print("\n--------------------------Test the best model-------------------------\n")

# test the best model
print("Train accuracy:", cv.score(X_train_sel_log, y_train_log))
print("Test accuracy:", cv.score(X_test_sel_log, y_test_log))

y_pred_log = cv.predict(X_test_sel_log)
print(classification_report(y_test_log, y_pred_log))

print("\n--------------------------DecisionTreeClassifier -------------------------\n")
from sklearn.tree import DecisionTreeClassifier

# similar parameters with the last practical
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2, 7),
          'min_samples_leaf': range(20, 60, 10)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train_log, y_train_log)

print(cv.best_params_)

from Task2_Stephen import analyse_feature_importance

# analyse feature importance from the tuned decision tree against log transformed X
analyse_feature_importance(cv.best_estimator_, X_log.columns)

from sklearn.feature_selection import SelectFromModel

# use the trained best decision tree from GridSearchCV to select features
# supply the prefit=True parameter to stop SelectFromModel to re-train the model
selectmodel = SelectFromModel(cv.best_estimator_, prefit=True)
X_train_sel_model = selectmodel.transform(X_train_log)
X_test_sel_model = selectmodel.transform(X_test_log)

print(X_train_sel_model.shape)

params = {'C': [pow(10, x) for x in range(-6, 4)]}

cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_sel_model, y_train_log)

print("Train accuracy:", cv.score(X_train_sel_model, y_train_log))
print("Test accuracy:", cv.score(X_test_sel_model, y_test_log))

# test the best model
y_pred = cv.predict(X_test_sel_model)
print(classification_report(y_test_log, y_pred))

# print parameters of the best model
print(cv.best_params_)





