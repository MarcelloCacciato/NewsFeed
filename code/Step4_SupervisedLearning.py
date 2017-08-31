
# coding: utf-8

# In[50]:

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
    
    print categories
 
    # Documents in a category
    #category_docs = reuters.fileids("money-supply")
 
    # Words for a document
    # document_id = category_docs[2]
    # document_words = reuters.words(category_docs[2])
    # print(document_words)  
 
    # Raw document
    # print(reuters.raw(document_id))


# In[51]:

collection_stats()


# In[52]:

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


# In[4]:

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


# Return the representer, without transforming
def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=0.01,
                        max_df=0.90, max_features=3000,
                        use_idf=True, sublinear_tf=True,
                        norm='l2');
    tfidf.fit(docs);
    return tfidf;

# max_df=0.9, max_features=200000,
# min_df=0.01, stop_words='english',
# use_idf=True, tokenizer=tokenize_and_stem, 
# ngram_range=(1,3)



# In[53]:

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=0.01,
                        max_df=0.90, max_features=3000,
                        use_idf=True, sublinear_tf=True,
                        norm='l2');


# In[5]:

def feature_values(doc, representer):
    doc_representation = representer.transform([doc])
    features = representer.get_feature_names()
    return [(features[index], doc_representation[0, index])
                 for index in doc_representation.nonzero()[1]]


# In[54]:

train_docs = []
test_docs = []
for doc_id in reuters.fileids():
    if doc_id.startswith("train"):
        train_docs.append(reuters.raw(doc_id))
    else:
        test_docs.append(reuters.raw(doc_id))


# In[55]:

print reuters.fileids()[0]


# In[8]:

#representer = tf_idf(train_docs);


# In[ ]:

#for doc in test_docs:
#    print(feature_values(doc, representer))


# In[56]:

vectorised_train_documents = tfidf.fit_transform(train_docs)


# In[57]:

from sklearn.metrics.pairwise import cosine_similarity
X = 1 - cosine_similarity(vectorised_train_documents)


# In[58]:

from sklearn.cluster import AffinityPropagation
from sklearn import metrics

# Compute Affinity Propagation
af = AffinityPropagation(damping=0.6).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='cosine'))
print("Calinski_Harabaz Coefficient: %0.3f"
      % metrics.calinski_harabaz_score(X, labels))


# In[ ]:

# it finds much more clusters than what we have (238 vs 90)
# Silhouette Coefficient is very low indicating that clusters overlap a lot


# In[ ]:

# selecting only 244 documents at random


# In[68]:

rndm_reuters_fileids_train = []
randint_train = []
#print random.uniform(0, len(rndm_reuters_fileids))
import numpy as np
for i in range(0,250):
    randint = np.random.randint(0,len(reuters.fileids()))
    randint_train.append(randint)
    rndm_reuters_fileids_train.append(reuters.fileids()[randint])
print rndm_reuters_fileids_train


# In[60]:

rndm_reuters_fileids_test = []
randint_test = []
#print random.uniform(0, len(rndm_reuters_fileids))
import numpy as np
for i in range(0,80):
    randint = np.random.randint(0,len(reuters.fileids()))
    randint_test.append(randint)
    rndm_reuters_fileids_test.append(reuters.fileids()[randint])
print rndm_reuters_fileids_test


# In[69]:

train_docs_160 = []
test_docs_80 = []
for doc_id in rndm_reuters_fileids_train:
    train_docs_160.append(reuters.raw(doc_id))
for doc_id in rndm_reuters_fileids_test:
    test_docs_80.append(reuters.raw(doc_id))


# In[62]:

vectorised_test_documents_80 = tfidf.transform(test_docs_80)


# In[70]:

vectorised_train_documents_160 = tfidf.transform(train_docs_160)


# In[71]:

from sklearn.metrics.pairwise import cosine_similarity
X = 1 - cosine_similarity(vectorised_train_documents_160)


# In[72]:

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


# In[ ]:




# In[ ]:




# In[ ]:




# In[12]:

vectorised_test_documents = tfidf.transform(test_docs)


# In[13]:

#List of document ids
documents = reuters.fileids()


# In[14]:

train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                            documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))


# In[15]:

# Transform multilabel labels
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id)
                                  for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id)
                             for doc_id in test_docs_id])


# In[ ]:



