
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

filepath_or_buffer = '../data/articles.csv'


# In[3]:

#import a data frame with 4 columns: orders, titles, descriptions, and articles
df = pd.read_csv(filepath_or_buffer)
num_of_articles=df.shape[0]
print "number of articles: ",num_of_articles


# In[4]:

header_df = list(df)
print header_df


# In[5]:

feed_article = df['feed_article']


# In[6]:

from bs4 import BeautifulSoup
feed_article_parsed = []
for text in feed_article:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    feed_article_parsed.append(text)


# In[7]:

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
import re
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


# In[8]:

import nltk
token_stems = []
for i in range(num_of_articles):
    token_stems.append(tokenize_and_stem(feed_article_parsed[i]))


# In[9]:

import csv
import ast
csvfile='tokenized_stems.csv'
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in token_stems:
        writer.writerow([val])    


# In[11]:

import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.01, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

get_ipython().magic(u'time tfidf_matrix = tfidf_vectorizer.fit_transform(feed_article_parsed)')
print(tfidf_matrix.shape)
num_articles = tfidf_matrix.shape[0]
print(num_articles)
num_terms = tfidf_matrix.shape[1]
print(num_terms)


# In[12]:

tfidf_mat = tfidf_matrix.toarray()


# In[13]:

import numpy as np

tfidf_df = pd.DataFrame(data=tfidf_mat.astype(float))
tfidf_df.to_csv('tfidf_mat.csv', sep=' ', header=False, float_format='%.5f', index=False)

