#!/usr/bin/env python
# coding: utf-8

# #### Requires a text file containing glove words and embeddings. Also requires passing in a set of words that you want embeddings of. Compare the desired set to output to see if any words were not embedded from your passed in set. If there is an issue embedding any words in the set that do appear in Glove, then a print statement will alert you to the words in question

# In[1]:


import numpy as np


# In[2]:


def main_load_glove_840B_300d(gloveFile, wordset):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        if word.lower() in wordset:
            try:
                embedding = np.asarray(splitLine[1:], dtype='float32')
                model[word] = embedding
            except:
                print("There was trouble parsing the word: ", word)
                pass
    print("Done.",len(model)," words loaded!")
    return model