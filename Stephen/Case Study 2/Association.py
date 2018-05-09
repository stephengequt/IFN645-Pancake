import pandas as pd
import numpy as np

df = pd.read_csv('datasets/pos_transactions.csv')
df.info()
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# Quantity_dist = sns.distplot(df['Quantity'].dropna())
# plt.show()

#Remove duplicates
df2 = df[['Transaction_Id','Product_Name']]
# df2.head(5)
df = df2.drop_duplicates()
df.head(5)


# group by account, then list all services
transactions = df.groupby(['Transaction_Id'])['Product_Name'].apply(list)

print(transactions.head(17))


from apyori import apriori

# type cast the transactions from pandas into normal list format and run apriori
transaction_list = list(transactions)
results = list(apriori(transaction_list, min_support=0.000000000000001))

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
print("##########################")


# sort all acquired rules descending by Confidence
result_df = result_df.sort_values(by='Confidence', ascending=False)
print(result_df.head(10))

import random
import matplotlib.pyplot as plt


support = result_df.as_matrix(columns=['Support'])
confidence = result_df.as_matrix(columns=['Confidence'])
#
# for i in range(len(support)):
#     support[i] = support[i] + 0.0025 * (random.randint(1, 10) - 5)
#     confidence[i] = confidence[i] + 0.0025 * (random.randint(1, 10) - 5)

plt.scatter(support, confidence, alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()




plt.plot(support, confidence,  marker='*')
plt.show()

# support
#
# plt.scatter(support,confidence, label='skitscat', color='k', s=25, marker="o")
#
# plt.xlabel('support')
# plt.ylabel('confidence')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()



pens_df = result_df[result_df['Left_side'] == "Pens"]
pens_df = pens_df.sort_values(by='Lift', ascending=False)
print(pens_df)
print(pens_df.info())