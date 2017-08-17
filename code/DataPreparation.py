
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


# In[7]:

#import a data frame with 4 columns: orders, titles, descriptions, and articles
df = pd.read_csv(filepath_or_buffer)
print "number of articles: ",df.shape[0]


# In[19]:

header_df = list(df)
print header_df


# In[27]:

for i in range(0,df.shape[1]):
    print len(df[str(header_df[i])]), header_df[i]
    if(len(df[str(header_df[i])]) != df.shape[0]):
        print "problem with number of rows in column",header_df[i]


# In[28]:

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')


# In[29]:

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# In[30]:

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


# In[71]:

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


# In[73]:

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


# In[77]:

#on entire feed article:
feed_vocab_frame = pd.DataFrame({'words': feed_vocab_tokenized}, index = feed_vocab_stemmed)
print(feed_vocab_frame)
#print feed_vocab_frame


# In[78]:

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.0, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

#%time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses_des)
get_ipython().magic(u'time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses_feed)')
print(tfidf_matrix.shape)


# In[79]:

terms = tfidf_vectorizer.get_feature_names()


# In[81]:

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print(dist)


# In[82]:

import matplotlib.pyplot as plt
# Display matrix
plt.matshow(dist.reshape((244, 244)))

plt.show()


# In[116]:

tfidf_mat = np.matrix(tfidf_matrix)
with open('outfile3.txt','wb') as f:
    for line in tfidf_mat:
        np.savetxt(f, line)
        #, fmt='%.2f')


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

