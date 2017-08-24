
# coding: utf-8

# In[1]:

import pandas as pd


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

feed_article = df['feed_article']


# In[6]:

from bs4 import BeautifulSoup
feed_article_clean = []
for text in feed_article:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    feed_article_clean.append(text)


# In[9]:

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

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


# In[36]:

import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.01, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

get_ipython().magic(u'time tfidf_matrix = tfidf_vectorizer.fit_transform(feed_article_clean)')
print(tfidf_matrix.shape)
num_articles = tfidf_matrix.shape[0]
print(num_articles)
num_terms = tfidf_matrix.shape[1]
print(num_terms)


# In[ ]:



