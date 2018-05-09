# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

import os

os.getcwd()
# read the veteran dataset
df = pd.read_csv('datasets/veteran.csv')

# show all columns information
print(df.info())

