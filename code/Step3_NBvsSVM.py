
# coding: utf-8

# In[1]:

import csv
import numpy as np
reader = csv.reader(open("tfidf_mat.csv", "rb"), delimiter=" ")
x = list(reader)
X = np.array(x).astype("float")


# In[3]:

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()


# In[4]:

predicted = clf.predict(X)


# In[ ]:



