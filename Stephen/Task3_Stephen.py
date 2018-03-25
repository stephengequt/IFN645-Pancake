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

from Task2 Stephen im