# load logs from wdata
wdata = open('Stephen/Prac/datasets/wdata.txt', 'r').readlines()

# print the first 3 lines
print('\n'.join(wdata[:4]))




import pandas as pd

# set names of pandas dataframe
names=['Host', 'Identd', 'Authuser', 'Date and time', 'Timezone', 'Request',
       'Status code', 'Bytes Sent', 'Referrer', 'Agent']
# read the dataframe
df = pd.read_csv('Stephen/Prac/datasets/wdata.txt', sep=' ', names=names, header=None)
# preview
df.head()

df.drop(0, inplace=True)  # drop the row with index 0, on axis 0 (row-wise)
# preview
df.head()

def extract_method_and_protocol(row):
    # function to extract HTTP request method and protocol from a request string
    request_splits = row['Request'].split()  # split request string by space
    row['Method'] = request_splits[0]
    row['Protocol'] = request_splits[-1]
    row['Request'] = ' '.join(request_splits[1:-1])  # stitch remaining request string back
    return row

df = df.apply(extract_method_and_protocol, axis=1)

# show the result
df.head()

df.info()


# correct the incorrect dataframe types
df['Status code'] = df['Status code'].astype(int)  # set status code to int
df['Datetime'] = pd.to_datetime(df['Date and time'], format='[%d/%b/%Y:%H:%M:%S')  # set date time to pandas datatime obj
df = df.drop(['Date and time'], axis=1)

# create a mask to filter all images
mask = (df['Request'].str.endswith('.gif') | df['Request'].str.endswith('.jpg') | df['Request'].str.endswith('.jpeg'))
print("# Rows before:", len(df))

# invert the mask, only keep records without .gif, .jpg and .jpeg in the request column
df2 = df[~mask]

print("After images removal", len(df2))

# second mask, remove all unsuccessful requests (code != 200)
df2 = df2[df2['Status code'] == 200]
print("After unsuccessful requests removal", len(df2))




from collections import defaultdict
import datetime

# first, make a copy of df2 just in case
df3 = df2.copy()

# sort the rows based on datetime, descending
df3.sort_values(by='Datetime', inplace=True)

# initiate session ID and user ID to 0
session_id = 0
user_id = 0

# create a dictionaries to hold last access information
last_access = defaultdict(lambda: datetime.datetime.utcfromtimestamp(0))

# dictionary to find previous session, user ID and steps assigned to a specific date/ip/browser key
session_dict = defaultdict(lambda: 1)
user_id_dict = defaultdict(lambda: 1)
session_steps = defaultdict(lambda: 1)


# function to be applied row wise
# for each row, produce session, user ID and path traversal
def get_log_user_info(row):
    # access global variables shared between all rows
    global session_id, user_id, session_dict, user_id_dict, session_steps, last_access

    session_key = str(row['Datetime'].date()) + '_' + row['Host']  # date + IP key for finding session
    user_key = str(row['Datetime'].date()) + '_' + row['Host'] + '_' + row[
        'Agent']  # date + IP + browser key for finding user
    time_diff_session = row['Datetime'] - last_access[session_key]  # session time diff
    time_diff_user = row['Datetime'] - last_access[user_key]  # user time diff

    # if the time diff from previous session is > 30 mins, assign new session ID
    if time_diff_session.total_seconds() > 1800:
        session_id += 1
        session_dict[session_key] = session_id

    # if the time diff from previous session is > 60 mins, assign new user ID
    if time_diff_user.total_seconds() > 3600:
        user_id += 1
        user_id_dict[user_key] = user_id

    # update last access for session and user
    last_access[session_key] = row['Datetime']
    last_access[user_key] = row['Datetime']

    # assign extracted info from the row
    row['Session'] = session_dict[session_key]
    row['Step'] = session_steps[row['Session']]
    row['User_ID'] = user_id_dict[user_key]
    session_steps[row['Session']] += 1
    return row


df3.head()

# apply function above to get a new dataframe with added information
df3 = df3.apply(get_log_user_info, axis=1)