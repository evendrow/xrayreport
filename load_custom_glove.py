#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[2]:


def main_load_custom_glove(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        try:
            embedding = np.asarray(splitLine[1:], dtype='float32')
            model[word] = embedding
        except:
            print("There was trouble parsing the word: ", word)
            pass
    print("Done.",len(model)," words loaded!")
    return model

