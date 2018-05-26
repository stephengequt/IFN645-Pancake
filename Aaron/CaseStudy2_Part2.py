#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:04:48 2018

@author: aaron
"""
import pandas as pd
import numpy as np

# load the pos transaction dataset
df = pd.read_csv('pos_transactions.csv')

# info and the first 10 transactions
print(df.info())
print(df.head(10))

# describe key statistics from Transaction Date and Product Name column
print(df['Transactin_Date'].describe())
print(df['Transactin_Date'].value_counts())

print(df['Product_Name'].describe())
print(df['Product_Name'].value_counts())


print ("-----------------------------------------")
print (" Number of missing value for each column ")
print ("-----------------------------------------")
459258 - (np.logical_not(df.isnull()).sum())

# group by account, then list all transaction
transactions = df.groupby(['Transaction_Id'])['Product_Name'].apply(list)

print(transactions.head(10))


from apyori import apriori

# type cast the transactions from pandas into normal list format and run apriori
transaction_list = list(transactions)
results = list(apriori(transaction_list, min_support=0.03))

# print first 5 rules
print(results[:5])

def convert_apriori_results_to_pandas_df(results):
    rules = []
    
    for rule_set in results:
        for rule in rule_set.ordered_statistics:
            # items_base = left side of rules, items_add = right side
            # support, confidence and lift for respective rules
            rules.append([','.join(rule.items_base), ','.join(rule.items_add),
                         rule_set.support, rule.confidence, rule.lift]) 
    
    # typecast it to pandas df
    return pd.DataFrame(rules, columns=['Left_side', 'Right_side', 'Support', 'Confidence', 'Lift']) 

result_df = convert_apriori_results_to_pandas_df(results)

print(result_df.head(20))


# sort all acquired rules descending by lift
result_df = result_df.sort_values(by='Lift', ascending=False)
print(result_df.head(10))

# sort all acquired rules descending by lift
result_df = result_df.sort_values(by='Confidence', ascending=False)
print(result_df.head(10))


###############plotting####################
support=result_df.as_matrix(columns=['Support'])
confidence=result_df.as_matrix(columns=['Confidence'])

import random
import matplotlib.pyplot as plt
 
for i in range (len(support)):
   support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
   confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
 
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('Support')
plt.ylabel('Confidence') 
plt.show()