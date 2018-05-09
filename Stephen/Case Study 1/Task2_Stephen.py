
import numpy as np
import pandas as pd
import pydot
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# Visualising
from io import StringIO
from sklearn import tree

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

#Default dataset
def data_prep():

    # read the organics dataset
    org1 = pd.read_csv('Organic_Clean.csv')

    #drop the unused target variable that is ORGANICS
    org1.drop(['ORGANICS','AGEGRP1'], axis=1, inplace=True)

    # one-hot encoding
    org1 = pd.get_dummies(org1)

    print(org1.info())

    return org1

def nodes_leaves(model):
    #The size of the tree
    n_nodes = model.tree_.node_count
    print(n_nodes)

    children_left = model.tree_.children_left
    children_right = model.tree_.children_right



    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes," % n_nodes)

    n_count = 0
    for i in range(n_nodes):
        if is_leaves[i]:
            n_count = n_count + 1
    print("and has %s leaf nodes." % n_count)


# call data_prep method
org1 = data_prep()

# target/input split
y = org1['ORGYN']
X = org1.drop(['ORGYN'], axis=1)

# setting random state
rs = 10

X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

# simple decision tree training
model = DecisionTreeClassifier(random_state=rs)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

nodes_leaves(model)

#analyze top 20 important features from the model
analyse_feature_importance(model, X.columns, n_to_display=20)

#number of nodes
print(model.tree_.node_count)

#number of leaves 
leave_id = model.apply(X_train)
leave_id = np.unique(leave_id)
print(len(leave_id))

# grab feature importances from the model and feature name from the original X
importances = model.feature_importances_
feature_names = X.columns

#variable is used for the first split? What are the competing splits for this first split
feature_names[model.tree_.feature]

#visualize
# visualize_decision_tree(model, X.columns, "stephentask2_org_vis.png")

print(" hyperparameters\n___________________________________________________________________________")
# hyperparameters and model performance
test_score = []
train_score = []

# check model performance for max depth from 2-20
for max_depth in range(2, 21):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=rs)
    model.fit(X_train, y_train)

    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))

# plot max depth hyperparameter values vs training and test accuracy score
plt.plot(range(2, 21), train_score, 'b', range(2,21), test_score, 'r')
plt.xlabel('max_depth\nBlue = Training Acc. Red = Test Acc.')
plt.ylabel('accuracy')
plt.show()




#grid search CV
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2,7),
          'min_samples_leaf': range(20,60,10)}

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
          'max_depth': range(2,6),
          'min_samples_leaf': range(35,56)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

#test the best model
y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

#print parameters of the best model
print(cv.best_params_)

model2 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=rs)
model2.fit(X_train, y_train)

print("Train accuracy:", model2.score(X_train, y_train))
print("Test accuracy:", model2.score(X_test, y_test))

y_pred2 = model2.predict(X_test)
print(classification_report(y_test, y_pred2))

nodes_leaves(model2)
