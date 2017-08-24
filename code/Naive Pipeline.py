
# coding: utf-8

# In[1]:

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


# In[5]:

twenty_train.target_names #prints all the categories
print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file


# In[6]:

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


# In[7]:

# TF: Just counting the number of words in each document has 1 issue: 
# it will give more weightage to longer documents than shorter documents. 
# To avoid this, we can use frequency (TF - Term Frequencies) i.e. #count(word) / #Total words, in each document.


# In[8]:

# TF-IDF: Finally, we can even reduce the weightage of more common words like (the, is, an etc.) which occurs in all document. 
# This is called as TF-IDF i.e Term Frequency times inverse document frequency.


# In[9]:

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[10]:

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


# In[11]:

#Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


# In[12]:

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)


# In[13]:

# Support Vector Machines (SVM): 
# Let’s try using a different algorithm SVM, 
# and see if we can get any better performance. 

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
np.mean(predicted_svm == twenty_test.target)


# In[14]:

# Almost all the classifiers will have various parameters which can be tuned to obtain optimal performance.
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3)}


# In[15]:

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)


# In[17]:

gs_clf.best_score_


# In[18]:

gs_clf.best_params_


# In[19]:

#from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)


# In[20]:

gs_clf_svm.best_score_


# In[21]:

gs_clf_svm.best_params_


# In[22]:

# Removing stop words: (the, then etc) from the data and not just downweighting them as TF-IDF does.
# You can expect some small improvement and not a huge one.
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(twenty_test.data)
print "This will probably be higher than 0.77: ",np.mean(predicted == twenty_test.target)


# In[23]:

text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
print "This will probably be higher than 0.82: ",np.mean(predicted_svm == twenty_test.target)


# In[24]:

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('mnb', MultinomialNB(fit_prior=False))])
text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)
np.mean(predicted_mnb_stemmed == twenty_test.target)


# In[25]:

filepath_or_buffer = '../data/articles.csv'


# In[26]:

import pandas as pd


# In[27]:

#import a data frame with 4 columns: orders, titles, descriptions, and articles
df = pd.read_csv(filepath_or_buffer)
print "number of articles: ",df.shape[0]


# In[28]:

header_df = list(df)
print header_df


# In[29]:

synopses_feed = df['feed_article']


# In[32]:

# using entire feed_article:
from bs4 import BeautifulSoup
synopses_clean = []
for text in synopses_feed:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean.append(text)
synopses_feed = synopses_clean 


# In[33]:

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



# In[40]:

import nltk
import re
#import os
#import codecs
#from sklearn import feature_extraction
#import mpld3
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.01, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

#%time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses_des)
get_ipython().magic(u'time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses_feed)')
print(tfidf_matrix.shape)


# In[ ]:



