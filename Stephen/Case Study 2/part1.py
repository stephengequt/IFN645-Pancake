######################################################################################
# Location Number         Numerical code for the store (unique identifier)
# DEALER CODE             Text identifier for the store(unique identifier)
# REPORT DATE             Date of the data extraction
# HATCH                   Number of hatch back model cars sold by the store
# SEDAN                   Number of sedan model cars sold by the store
# WAGON                   Number of station wagon model cars sold by the store
# UTE                     Number of utility / tray back model cars sold by the store
# K_SALES_TOT             Total sales for the store($$$)
######################################################################################


######################################################################################
#Input dataset
import pandas as pd
import numpy as np

df = pd.read_csv('Stephen/Case Study 2/datasets/model_car_sales.csv', na_filter=False)
df.info()

# get more information from RegDens
print(df['HATCH'].describe())
print(df['HATCH'].value_counts().nlargest(5))

# replace the empty strings in the series with nan and typecast to float
df['HATCH'] = df['HATCH'].replace('', np.nan).astype(float)
df['WAG0N'] = df['WAG0N'].replace('', np.nan).astype(float)
df['SEDAN'] = df['SEDAN'].replace('', np.nan).astype(float)
df['UTE'] = df['UTE'].replace('', np.nan).astype(float)
df.info()

print ("-----------------------------------------")
print (" Number of missing value for each column ")
print ("-----------------------------------------")
675 - (np.logical_not(df.isnull()).sum())


import seaborn as sns
import matplotlib.pyplot as plt

UTE_dist = sns.distplot(df['UTE'].dropna())
plt.show()

HATCH_dist = sns.distplot(df['HATCH'].dropna())
plt.show()

WAG0N_dist = sns.distplot(df['WAG0N'].dropna())
plt.show()

SEDAN_dist = sns.distplot(df['SEDAN'].dropna())
plt.show()

print(df['UTE'].describe())
print(df['HATCH'].describe())
print(df['WAG0N'].describe())
print(df['SEDAN'].describe())

# # create a mask of errorneous MeanHHSz values
# df['Missing_UTE'] = (df['UTE'] == 0)
#
# # use FaceTGrid to plot the distribution of MedHHInc when MeanHHSZ is errorneous
# g = sns.FacetGrid(df, col='Missing_UTE')
# g = g.map(plt.hist, 'HATCH', bins=100)
#
# plt.show()



# before
print("Row # before dropping errorneous rows", len(df))

# a very easy way to drop rows with MeanHHSz values below 1
df = df[df['UTE'].notnull()]

# after
print("Row # after dropping errorneous rows", len(df))


################Default Model#####################
from sklearn.preprocessing import StandardScaler



# take 3 variables and drop the rest
df2 = df[['HATCH', 'WAG0N', 'SEDAN']]

# convert df2 to matrix
X = df2.as_matrix()
X
# # scaling
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X

from sklearn.cluster import KMeans

# random state, we will use 42 instead of 10 for a change
rs = 42

# set the random state. different random state seeds might result in different centroids locations
model = KMeans(n_clusters=3, random_state=rs)
model.fit(X)

# sum of intra-cluster distances
print("Sum of intra-cluster distance:", model.inertia_)

print("Centroid locations:")
for centroid in model.cluster_centers_:
    print(centroid)

model = KMeans(n_clusters=3, random_state=rs).fit(X)

# assign cluster ID to each record in X
# Ignore the warning, does not apply to our case here
y = model.predict(X)
df2['Cluster_ID'] = y

# how many records are in each cluster
print("Cluster membership")
print(df2['Cluster_ID'].value_counts())

# pairplot the cluster distribution.
cluster_g = sns.pairplot(df2, hue='Cluster_ID')
plt.show()


###########Standardised Model###############
# convert df2 to matrix
Y = df2.as_matrix()
Y
# # scaling
scaler = StandardScaler()
Y = scaler.fit_transform(X)

from sklearn.cluster import KMeans

# random state, we will use 42 instead of 10 for a change
rs = 42

# set the random state. different random state seeds might result in different centroids locations
model = KMeans(n_clusters=3, random_state=rs)
model.fit(Y)

# sum of intra-cluster distances
print("Sum of intra-cluster distance:", model.inertia_)

print("Centroid locations:")
for centroid in model.cluster_centers_:
    print(centroid)

model = KMeans(n_clusters=3, random_state=rs).fit(X)

# assign cluster ID to each record in X
# Ignore the warning, does not apply to our case here
y = model.predict(Y)
df2['Cluster_ID'] = y

# how many records are in each cluster
print("Cluster membership")
print(df2['Cluster_ID'].value_counts())

# pairplot the cluster distribution.
cluster_g = sns.pairplot(df2, hue='Cluster_ID')
plt.show()

############# Elbow ################
# list to save the clusters and cost
clusters = []
inertia_vals = []

# this whole process should take a while
for k in range(2, 15, 2):
    # train clustering with the specified K
    model = KMeans(n_clusters=k, random_state=rs, n_jobs=10)
    model.fit(Y)
    # append model to cluster list
    clusters.append(model)
    inertia_vals.append(model.inertia_)
    
# plot the inertia vs K values
plt.plot(range(2,15,2), inertia_vals, marker='*')
plt.show()

####### silhouette_score

from sklearn.metrics import silhouette_score

print(clusters[1])
print("Silhouette score for k=4", silhouette_score(X, clusters[1].predict(X)))

print(clusters[2])
print("Silhouette score for k=6", silhouette_score(X, clusters[2].predict(X)))




# visualisation of K=4 clustering solution
model = KMeans(n_clusters=6, random_state=rs)
model.fit(Y)

# sum of intra-cluster distances
print("Sum of intra-cluster distance:", model.inertia_)

print("Centroid locations:")
for centroid in model.cluster_centers_:
    print(centroid)

y = model.predict(Y)
df2['Cluster_ID'] = y

# how many in each
print("Cluster membership")
print(df2['Cluster_ID'].value_counts())

# pairplot
# added alpha value to assist with overlapping points
cluster_g = sns.pairplot(df2, hue='Cluster_ID', plot_kws={'alpha': 0.5})
plt.show()