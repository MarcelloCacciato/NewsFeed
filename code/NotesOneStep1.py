# to run after Step1_*
print tfidf_vectorizer.vocabulary_
words = [tfidf_vectorizer.vocabulary_.keys()[i] for i in range(num_terms)]
#print words
test = [tfidf_vectorizer.vocabulary_.values()[i] for i in range(num_terms)]
#print test
import numpy as np
indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
features = tfidf_vectorizer.get_feature_names()
top_n = num_terms/10
top_features = [features[i] for i in indices[:top_n]]
print top_features
# how comes that many terms are actually more than one word?
# isn't strange that sometimes a term is 3 words and the previous one was the first two words of these three? 
# (e.g. u'order lock', u'order lock door')


