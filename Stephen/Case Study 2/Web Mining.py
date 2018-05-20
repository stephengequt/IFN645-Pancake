import pandas as pd
import numpy as np

df = pd.read_csv('Stephen/Case Study 2/datasets/web_log_data.csv', na_filter=False)

df.info()
#
#get more information form RegDens
# print(df['request'].describe())
print(df['request'].value_counts())

df['date_time'] = pd.to_datetime(df['date_time'], format="%d/%b/%Y:%H:%M:%S")  # set date time to pandas datatime obj
# df.info()
# print(df['request'].describe())
df['request'] = df['request'].str.rstrip('\/')
df.replace(r'^\s*$', 'homepage', regex=True, inplace = True)
# df.info()
# print(df['request'].describe())
#
# print(df['user_id'].describe())
# print(df['user_id'].value_counts())

# group by user_id, then list all requests

df['Day'] = df['date_time'].dt.day
df['Month'] = df['date_time'].dt.month
df['Hour'] = df['date_time'].dt.hour
df['Day_of_week'] = df['date_time'].dt.dayofweek
df.info()

user1 = df[df['user_id'] == 1]
print(user1)


requestsByUser = df.groupby(['user_id'])['request'].apply(list)

print(requestsByUser.head(5))


from apyori import apriori

# type cast the transactions from pandas into normal list format and run apriori
requestsByUser_list = list(requestsByUser)
results = list(apriori(requestsByUser_list, min_support=0.1))

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

result_df = result_df.sort_values(by='Confidence', ascending=False)
print(result_df.head(10))
#
# requestsByUser = df.groupby(['request'])['Hour'].apply(list)
#
# print(requestsByUser.head(5))
#
#
# from apyori import apriori
#
# # type cast the transactions from pandas into normal list format and run apriori
# requestsByUser_list = list(requestsByUser)
# results = list(apriori(requestsByUser_list, min_support=0.3))
#
# # print first 5 rules
# print(results[:5])
#
#
# def convert_apriori_results_to_pandas_df(results):
#     rules = []
#
#     for rule_set in results:
#         for rule in rule_set.ordered_statistics:
#             # items_base = left side of rules, items_add = right side
#             # support, confidence and lift for respective rules
#             rules.append([','.join(rule.items_base), ','.join(rule.items_add),
#                           rule_set.support, rule.confidence, rule.lift])
#
#             # typecast it to pandas df
#     return pd.DataFrame(rules, columns=['Left_side', 'Right_side', 'Support', 'Confidence', 'Lift'])
#
#
# result_df = convert_apriori_results_to_pandas_df(results)
#
# print(result_df.head(20))
#
# # sort all acquired rules descending by lift
# result_df = result_df.sort_values(by='Lift', ascending=False)
# print(result_df.head(10))



############################################################################
#Classification#
# 1.