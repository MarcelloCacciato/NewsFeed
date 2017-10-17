
# coding: utf-8

# In[5]:

import csv
import numpy as np
reader = csv.reader(open("tfidf_mat.csv", "rb"), delimiter=" ")
x = list(reader)
xx = np.array(x).astype("float")
from sklearn.metrics.pairwise import cosine_similarity
X = 1 - cosine_similarity(xx)


# In[6]:

from sklearn.cluster import AffinityPropagation
from sklearn import metrics

# Compute Affinity Propagation
for damp in np.arange(0.5,1.0,0.05):
    af = AffinityPropagation(damping=damp).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)
    print('Damping:',damp)
    print('   Estimated number of clusters: %d' % n_clusters_)
    print("   Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, labels, metric='cosine'))
    print("   Calinski_Harabaz Coefficient: %0.3f"
        % metrics.calinski_harabaz_score(X, labels))


# In[7]:

import collections
a = labels
counter=collections.Counter(a)
print(counter)
print(counter.values())
print(counter.keys())
print(counter.most_common(33))


# In[ ]:



