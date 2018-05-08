#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:23:26 2018

@author: Steve
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

os.getcwd()

df = pd.read_csv('dataset/organics.csv')

print(df.info())

#Convert EDATE and DOB to Date-Time Format
DOB_Object = [datetime.strptime(x,'%Y-%m-%d')for x in df['DOB']]
eDate = datetime.strptime(df['EDATE'][0], "%Y-%m-%d")
AGE = list()
#Get age by EVA.YEAR-DOB.YEAR -((DOB.MONTH, DOB.DAY) < (EVA.MONTH, EVA.DAY))
for dob in DOB_Object:
    # for each dob in DOB_Object
    age = eDate.year - dob.year - ((dob.month, dob.day)>(eDate.month, eDate.day))
    AGE.append(age)
df['AGE'] = AGE

#Filled up AGEGRP1 and AGEGRP2 ######
condlist = [df['AGE']<=20, df['AGE']<=40, df['AGE']<=60, df['AGE']<=80]
choicelist = ['<20','20-40','40-60','60-80']
df['AGEGRP1'] = np.select(condlist, choicelist)

condlist1 = [df['AGE']<=20, df['AGE']<=30, df['AGE']<=40, df['AGE']<=50, df['AGE']<=60, df['AGE']<=70, df['AGE']<=80]
choicelist1 = ['10-20','20-30','30-40','40-50','50-60','60-70','70-80']
df['AGEGRP2'] = np.select(condlist1, choicelist1)

# Filled up missing value in GENDER
df['GENDER'].fillna('U',inplace = True)

# Impute LTIME with median
df['LTIME'].fillna(df['LTIME'].median(), inplace=True)

# Impute AFFL with median/mean (both have a close similar value) ######

df['AFFL'].fillna(df['AFFL'].median(), inplace=True)

# Impute REGION with 'UNKNOWN'
df['REGION'].fillna('Unknown',inplace = True)

# Impute NGROUP with 'UNKNOWN'
df['NGROUP'].fillna('U',inplace = True)

#Drop variables that are not needed
df.drop(['DOB', 'EDATE','LCDATE','AGE','CUSTID','NEIGHBORHOOD', 'TV_REG','BILL'], axis=1, inplace=True)
df.drop(['AGEGRP1','ORGANICS'], axis=1, inplace=True)

#Filled up AGEGRP1 and AGEGRP2 ######
condlist = [df['LTIME']<=10, df['LTIME']<=20, df['LTIME']<=30, df['LTIME']<=40]
choicelist = ['<10','10-20','20-30','30-40']
df['LTIME'] = np.select(condlist, choicelist)

print(df['LTIME'])

# one-hot encoding
df = pd.get_dummies(df)

print(df.info())

df.to_csv('VeryClean.csv')