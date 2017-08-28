
# coding: utf-8

# In[ ]:




# In[38]:

import nltk
import string

from collections import Counter

def get_tokens():
   with open('tokenized_stems.csv', 'r') as shakes:
    text = shakes.read()
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    no_punctuation = lowers.translate(None, string.punctuation)
    no_apos = no_punctuation.translate(None, "'")
    tokens = nltk.word_tokenize(no_apos)
    return tokens

tokens = get_tokens()
count = Counter(tokens)
for i in range(115771):
    tokens[i]=tokens[i][1:]
count2 = Counter(tokens)
#print count.most_common(10)
print count2.most_common(10)


# In[39]:

from nltk.corpus import stopwords

filtered = [w for w in tokens if not w in stopwords.words('english')]
count = Counter(filtered)
print count.most_common(10)

