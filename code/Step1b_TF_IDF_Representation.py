
# coding: utf-8

# In[ ]:

''''''


# In[ ]:

'''loading tf_idf matrix and showing distance among documents'''


# In[ ]:

''''''


# In[16]:

import csv
import numpy as np
reader = csv.reader(open("tfidf_mat.csv", "rb"), delimiter=" ")
x = list(reader)
X = np.array(x).astype("float")


# In[17]:

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(X)


# In[18]:

import matplotlib.pyplot as plt
from matplotlib import cm
# Make plot with horizontal colorbar
fig, ax = plt.subplots()

data = dist.reshape((244, 244))

cax = ax.imshow(data, interpolation='nearest', cmap=cm.YlOrRd)#cm.afmhot)
ax.set_title('Distance via Cosine Similarity')

cbar = fig.colorbar(cax, ticks=[0, 0.5, 1], orientation='vertical')
cbar.ax.set_yticklabels(['Small', 'Medium', 'Large'])  

plt.savefig('cosine_similarity.png')
plt.show()


# In[19]:

from sklearn.metrics.pairwise import euclidean_distances

dist_eucl = euclidean_distances(X,squared=True)/2.

plt.matshow(dist_eucl.reshape((244, 244)))
plt.savefig('euclidean_distance.png')
plt.show()

