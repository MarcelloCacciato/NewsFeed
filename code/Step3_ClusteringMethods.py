
# coding: utf-8

# In[18]:

import csv
import numpy as np
reader = csv.reader(open("tfidf_mat.csv", "rb"), delimiter=" ")
x = list(reader)
xx = np.array(x).astype("float")
from sklearn.metrics.pairwise import cosine_similarity
X = 1 - cosine_similarity(xx)


# In[28]:

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


# In[22]:

damp=0.6
af = AffinityPropagation(damping=damp).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print labels


# In[27]:

import collections
a = labels
counter=collections.Counter(a)
print(counter)
# Counter({1: 4, 2: 4, 3: 2, 5: 2, 4: 1})
print(counter.values())
# [4, 4, 2, 1, 2]
print(counter.keys())
# [1, 2, 3, 4, 5]
print(counter.most_common(33))
# [(1, 4), (2, 4), (3, 2)]


# In[ ]:




# In[3]:

from sklearn.cluster import MeanShift, estimate_bandwidth
# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X)
#, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


# In[ ]:

from sklearn.cluster import spectral_clustering


# In[7]:

from sklearn import cluster, datasets, mixture

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3}

params = default_base.copy()
# ============
# Create cluster objects
# ============
    # estimate bandwidth for mean shift
params['quantile'] = 0.3
bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    #ward = cluster.AgglomerativeClustering(
        #n_clusters=params['n_clusters'], linkage='ward',
        #connectivity=connectivity)
    #spectral = cluster.SpectralClustering(
        #n_clusters=params['n_clusters'], eigen_solver='arpack',
        #affinity="nearest_neighbors")
params['eps']=0.3
dbscan = cluster.DBSCAN(eps=params['eps'])
    
params['damping']=0.9
params['preference'] = -200
affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
    #average_linkage = cluster.AgglomerativeClustering(
        #linkage="average", affinity="cityblock",
        #n_clusters=params['n_clusters'], connectivity=connectivity)
    #birch = cluster.Birch(n_clusters=params['n_clusters'])
    #gmm = mixture.GaussianMixture(
        #n_components=params['n_clusters'], covariance_type='full')

clustering_algorithms = (
        #('MiniBatchKMeans', two_means),
    ('AffinityPropagation', affinity_propagation),
    ('MeanShift', ms)
        #,
        #('SpectralClustering', spectral),
        #('Ward', ward),
        #('AgglomerativeClustering', average_linkage),
        #('DBSCAN', dbscan)
        #,
        #('Birch', birch),
        #('GaussianMixture', gmm)
)
for name, algorithm in clustering_algorithms:
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, y_pred, metric='cosine'))


# In[22]:

db = DBSCAN(eps=0.3, min_samples=10).fit(X)

