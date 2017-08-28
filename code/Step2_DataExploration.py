
# coding: utf-8

# In[6]:

import csv
import numpy as np
reader = csv.reader(open("tfidf_mat.csv", "rb"), delimiter=" ")
x = list(reader)
X = np.array(x).astype("float")


# In[7]:

# k means determine k
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
distortions = []
K = [10,20,30,40,50]
#range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


# In[10]:

K = [60,70,80,90,100]
#range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


# In[11]:

K=[10,20,30,40,50,60,70,80,90,100]


# In[12]:

import matplotlib.pyplot as plt
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[13]:

#Clearly The Elbow Method using the distorsion does not work well with the data set we have
# we try by changing the metric from euclidean to cosine


# In[14]:

K = [10,20,30,40,50,60,70,80,90,100]
distortionsCosine = []
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortionsCosine.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'cosine'), axis=1)) / X.shape[0])


# In[15]:

def PlotTheElbow(K,disto):
    import matplotlib.pyplot as plt
    # Plot the elbow
    plt.plot(K, disto, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


# In[16]:

PlotTheElbow(K,distortionsCosine)


# In[ ]:

# Elbow still not very evident, we will move to different classifiers

