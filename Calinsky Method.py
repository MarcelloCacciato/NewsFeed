
# coding: utf-8

# In[1]:

from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets


# In[2]:

dataset = datasets.load_iris()
X = dataset.data
y = dataset.target


# In[6]:

import pandas as pd
filepath_or_buffer = '../data/articles.csv'
#import a data frame with 4 columns: orders, titles, descriptions, and articles
df = pd.read_csv(filepath_or_buffer)
print "number of articles: ",df.shape[0]

X = df


# In[7]:

import numpy as np
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.calinski_harabaz_score(X, labels)
for k in range(2, 21):
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X)
    labels = kmeans_model.labels_
    labels = kmeans_model.labels_
    print k, metrics.calinski_harabaz_score(X, labels)

