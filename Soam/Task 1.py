# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:51:16 2018

@author: Soam wei jie
"""

import pandas as pd
import os

os.getcwd()

df = pd.read_csv('organics.csv')

print(df.info())

print(df['GENDER'].describe())

print(df['AGE'].describe())

print(df['AGEGRP1'].describe())

print(df['AGEGRP2'].describe())

print(df['GENDER'].value_counts())

print(df['AGE'].value_counts())

******************************************************************************************

print("Raw count of genders of customers with organics purchased")
print(df.groupby(['ORGYN'])['GENDER'].value_counts())

print("------------------")

# add normalisation to get the relative frequency
print("Normalised count (percentage) of genders of customers with organics purchased")
print(df.groupby(['ORGYN'])['GENDER'].value_counts(normalize=True))


******************************************************************************************


print("Raw count of AGEGRP of customers with organics purchased")
print(df.groupby(['ORGYN'])['AGEGRP2'].value_counts())

print("------------------")

# add normalisation to get the relative frequency
print("Normalised count (percentage) of AGEGRP of customers with organics purchased")
print(df.groupby(['ORGYN'])['AGEGRP2'].value_counts(normalize=True))


*******************************************************************************************


print("Raw count of NGROUP of customers with organics purchased")
print(df.groupby(['ORGYN','AGEGRP2','GENDER'])['NGROUP'].value_counts())

print("------------------")

# add normalisation to get the relative frequency
print("Normalised count (percentage) of NGROUP of customers with organics purchased")
print(df.groupby(['ORGYN','AGEGRP2','GENDER'])['NGROUP'].value_counts(normalize=True))

*******************************************************************************************

print("Raw count of CLASS of customers with organics purchased")
print(df.groupby(['ORGANICS','ORGYN'])['CLASS'].value_counts())

print("------------------")

# add normalisation to get the relative frequency
print("Normalised count (percentage) of CLASS of customers with organics purchased")
print(df.groupby(['ORGANICS','ORGYN'])['CLASS'].value_counts(normalize=True))

*******************************************************************************************


import matplotlib.pyplot as plt
import seaborn as sns

dg = sns.distplot(df['AGE'].dropna())
plt.show()


print("Raw count of genders of customers with organics purchased")
print(df.groupby(['ORGYN'])['AGEGRP2'].value_counts())

print("------------------")

# add normalisation to get the relative frequency
print("Normalised count (percentage) of genders of customers with organics purchased")
print(df.groupby(['ORGYN'])['AGEGRP2'].value_counts(normalize=True))


dg = sns.distplot(df['AGEGRP2'].dropna())
plt.show()

dg = sns.countplot(data=df, x='ORGYN')
plt.show()

ax = sns.boxplot(x="ORGYN", y="LTIME", data=df)
plt.show()
