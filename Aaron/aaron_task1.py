#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:15:36 2018

@author: Aaron Lim n8021911
"""
import os

os.getcwd()

import numpy as np
import pandas as pd

def data_prep():
    
    # read the organics dataset 
    org = pd.read_csv('organics.csv')

    # drop unnecesary variables
    org.drop(['CUSTID'], axis=1, inplace=True)
    
    # remove data with NaN
    org = org.dropna(axis = 0, how='any')
    
    print(org.info())
    
    return org

# run data_prep method
org = data_prep()

# get the average age of buyers, grouped by their number of purchases
print(org.groupby(['ORGANICS'])['AGE'].mean())

# get the value count of each gender
print("Raw count of gender organic buying habits")
print(org.groupby(['ORGANICS'])['GENDER'].value_counts())

print("------------------")

# normalisation to get the relative frequency
print("Normalised count (percentage) of gender organic buying habits")
print(org.groupby(['ORGANICS'])['GENDER'].value_counts(normalize=True))


# Task1-1
    
import matplotlib.pyplot as plt
import seaborn as sns

# histograms
orgp = sns.distplot(org['AGE'].dropna())
plt.show()

orgp = sns.countplot(data=org, x='GENDER')
plt.show()

orgp = sns.countplot(data=org, x='AGEGRP1')
plt.show()

orgp = sns.countplot(data=org, x='AGEGRP2')
# plt.xticks(range(len(orgp)), ('10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80'))
plt.show()

orgp = sns.countplot(data=org, x='TV_REG')
plt.xticks(rotation=90)
plt.show()

# boxplots
ax = sns.boxplot(x="GENDER", y="ORGANICS", data=org)
plt.show()

ax = sns.boxplot(x="GENDER", y="AGE", data=org)
plt.show()

ax = sns.boxplot(x="GENDER", y="BILL", data=org)
plt.show()


# Task1-2
    
# one-hot encoding
org = pd.get_dummies(org)

print (org['ORGYN'].value_counts())

orgp = sns.countplot(data=org, x='ORGYN')
plt.show()
