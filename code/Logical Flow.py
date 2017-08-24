
# coding: utf-8

# In[1]:

# Logical flow of the minimal algorithm I want to present

# Assignment: 
# Our client wants to build a mechanism where similar articles are grouped together based on their relevancy.
# So for a corpus of articles, device a clustering model that groups similar articles together. 
# Given that the volume of the news is going to increase manifold, make sure that the solution is scalable.


# In[ ]:

# Main Approach: clustering based on the tf_idf matrix and SVM


# In[ ]:

'''
   # mention and show how you clean the data set 
   # before feeding it to tf_idf (tokenizer, stemmer)
   # comment on the tf_idf parameters (max_df, max_features,min_df=0.01, stop_words='english', ngram_range)


# In[ ]:

'''
   # mention and show why you picked SVM
   # perform a grid search where you assess how some relevant metrics change as parameters changes


# In[ ]:

'''
   # show the main result: classification of the articles given the choice made in the steps above
   # show a few examples of the articles that belong to different groups and see if they really differ


# In[ ]:

# Supplementary Material:


# In[ ]:

'''
   1 - show results obtained using titles only, description only, text only, (and all three of them ?)
   [find a way to compare classifications] 


# In[ ]:

'''
   2 - sort of scalability test. Apply the main analysis to 100, 150 and 244 (all) articles 


# In[ ]:

# Discussion and Future Improvements


# In[ ]:

'''
   A - use a large dataset (~10'000 articles) to apply supervised ml using the Reuters dataset 


# In[ ]:

'''
   B - use the Reuters dataset ignoring the classification, apply main method to 250, 2500, 5000, 10'000 articles 
   how does the algorithm behave? is the number of categories robust? 
   if I impose to Reuters the same number of categories as I found in my dataset, 
   do articles in similar groups actually talk about similar topics ?  


# In[ ]:

'''
   C - mention latent dirichlet and multi-classification


# In[ ]:

# use SUPERVISED MACHINE LEARNING


# In[ ]:

# use reuters corpora for training (and testing) purpose
# 10788 documents
# 7769 total train documents
# 3019 total test documents
# 90 categories
    # group categories somehow (remember the ydatalytics dataset is only 244 articles and 90 categories might be too much)
    # 


# In[ ]:

# apply tokenizer and stemmer


# In[ ]:

#create tf_idf matrix


# In[ ]:

# run a classifier on the train dataset, look at th results in terms of the f1_score,precision_score,recall_score


# In[ ]:

# run a grid search on the parameters of the classifier used above 
# and see if f1_score,precision_score,recall_score improve

