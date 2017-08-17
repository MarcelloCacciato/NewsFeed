
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3


# In[2]:

filepath_or_buffer = '../data/articles.csv'


# In[3]:

#import a data frame with 4 columns: orders, titles, descriptions, and articles
df = pd.read_csv(filepath_or_buffer)
print "number of articles: ",df.shape[0]


# In[4]:

header_df = list(df)
print header_df


# In[5]:

for i in range(0,df.shape[1]):
    print len(df[str(header_df[i])]), header_df[i]
    if(len(df[str(header_df[i])]) != df.shape[0]):
        print "problem with number of rows in column",header_df[i]


# In[6]:

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')


# In[7]:

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[8]:

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[9]:

synopses_des = df['description']
synopses_feed = df['feed_article']
synopses_clean = []
'''
using only description:
for text in synopses_des:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean.append(text)
synopses_des = synopses_clean
'''
# using entire feed_article:
for text in synopses_feed:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean.append(text)
synopses_feed = synopses_clean 


# In[10]:

'''
# I do this analyses on the description only!
des_vocab_stemmed = []
des_vocab_tokenized = []
for i in synopses_des:
    des_words_stemmed = tokenize_and_stem(i)
    des_vocab_stemmed.extend(des_words_stemmed)
    
    des_words_tokenized = tokenize_only(i)
    des_vocab_tokenized.extend(des_words_tokenized)
'''
#while this is done on the entire feed article
feed_vocab_stemmed = []
feed_vocab_tokenized = []
for i in synopses_feed:
    feed_words_stemmed = tokenize_and_stem(i)
    feed_vocab_stemmed.extend(feed_words_stemmed)
    
    feed_words_tokenized = tokenize_only(i)
    feed_vocab_tokenized.extend(feed_words_tokenized)


# In[35]:

#only on description:
#des_vocab_frame = pd.DataFrame({'words': des_vocab_tokenized}, index = des_vocab_stemmed)
#print des_vocab_frame


# In[11]:

#on entire feed article:
feed_vocab_frame = pd.DataFrame({'words': feed_vocab_tokenized}, index = feed_vocab_stemmed)
print(feed_vocab_frame)
#print feed_vocab_frame


# In[12]:

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.0, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

#%time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses_des)
get_ipython().magic(u'time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses_feed)')
print(tfidf_matrix.shape)


# In[13]:

terms = tfidf_vectorizer.get_feature_names()


# In[14]:

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print(dist)


# In[15]:

import matplotlib.pyplot as plt
# Display matrix
plt.matshow(dist.reshape((244, 244)))

plt.show()


# In[16]:

tfidf_mat = tfidf_matrix.toarray()
print tfidf_mat


# In[19]:

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = tfidf_mat
for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))


# In[20]:

for n_cluster in range(11, 21):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))


# In[22]:

# k means determine k
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


# In[23]:

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[24]:

K = range(10,21)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


# In[26]:

K = range(1,21)
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[31]:

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)


# In[32]:

ks = range(1,10)


# In[34]:

from sklearn import cluster
# run 9 times kmeans and save each result in the KMeans object
KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]


# In[36]:

from scipy.spatial import distance
# now run for each cluster the BIC computation
BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]

print BIC


# In[42]:

import numpy as np
#from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

#kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
#labels = kmeans_model.labels_
#metrics.calinski_harabaz_score(X, labels)
for count in range(2, 21):
    kmeans_mod = KMeans(n_clusters=count, random_state=1).fit(X)
    lab = kmeans_mod.labels_
    #labels = kmeans_model.labels_
    print count, metrics.calinski_harabaz_score(X, lab)


# In[83]:

from sklearn.cluster import KMeans

num_clusters = 10

km = KMeans(n_clusters=num_clusters)

get_ipython().magic(u'time km.fit(tfidf_matrix)')

clusters = km.labels_.tolist()
print(clusters)


# In[84]:

from sklearn.externals import joblib


# In[85]:

joblib.dump(km,  'feed_cluster.pkl')


# In[86]:

#description only:
#km = joblib.load('des_cluster.pkl')
km = joblib.load('feed_cluster.pkl')
clusters = km.labels_.tolist()


# In[87]:

titles = df['title'].tolist()
ranks = df['Unnamed: 0'].tolist()
synopses = df['description'].tolist()
articles = df['feed_article'].tolist()

feeds = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'article': articles,'cluster': clusters}

frame = pd.DataFrame(feeds, index = [clusters] , columns = ['rank', 'cluster', 'title', 'synopsis','article'])


# In[89]:

frame['cluster'].value_counts()


# In[90]:

grouped = frame['rank'].groupby(frame['cluster'])

grouped.mean()


# In[93]:

from __future__ import print_function

print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]:
        #print(' %s' % des_vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        print(' %s' % feed_vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()
    print()

