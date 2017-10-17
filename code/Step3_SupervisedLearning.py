
# coding: utf-8

# In[1]:

from nltk.corpus import reuters 
 
def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents")
 
    train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents))
    
    test_docs = list(filter(lambda doc: doc.startswith("test"),
                       documents))
    
    # List of categories
    categories = reuters.categories()
    print(str(len(categories)) + " categories")


# In[2]:

collection_stats()


# In[3]:

#select 250 random documents
rndm_reuters_fileids_train = []
randint_train = []
#print random.uniform(0, len(rndm_reuters_fileids))
import numpy as np
for i in range(0,250):
    randint = np.random.randint(0,len(reuters.fileids()))
    randint_train.append(randint)
    rndm_reuters_fileids_train.append(reuters.fileids()[randint])


# In[4]:

train_docs_250 = []
for doc_id in rndm_reuters_fileids_train:
    train_docs_250.append(reuters.raw(doc_id))


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

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=0.01,
                        max_df=0.90, max_features=3000,
                        use_idf=True, sublinear_tf=True,
                        norm='l2');


# In[11]:

import nltk
vectorised_train_documents_250 = tfidf.fit_transform(train_docs_250)


# In[12]:

from sklearn.metrics.pairwise import cosine_similarity
X = 1 - cosine_similarity(vectorised_train_documents_250)


# In[13]:

from sklearn.cluster import AffinityPropagation
from sklearn import metrics

# Compute Affinity Propagation
for damp in np.arange(0.5,1.0,0.05):
    af = AffinityPropagation(damping=damp).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)
    print('Damping: ',damp)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, labels, metric='cosine'))
    print("Calinski_Harabaz Coefficient: %0.3f"
        % metrics.calinski_harabaz_score(X, labels))

