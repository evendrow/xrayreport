#!/usr/bin/env python
# coding: utf-8

# In[1]:


from load_glove_840B_300d import *
from make_study_dictionary import *
from load_custom_glove import *


# In[2]:


def load_session_dictionary():
    train_dict = main_make_study_dictionary("train")
    val_dict = main_make_study_dictionary("val")
    test_dict = main_make_study_dictionary("test")
    
    return train_dict, val_dict, test_dict


# In[3]:


def build_corpus(train_dict, val_dict, test_dict):
    corpus = []
    for visit in train_dict.keys():
        clinical_note = train_dict[visit]["clinical_note"]
        corpus.append(clinical_note)
    
    for visit in val_dict.keys():
        clinical_note = val_dict[visit]["clinical_note"]
        corpus.append(clinical_note)
    
    for visit in test_dict.keys():
        clinical_note = test_dict[visit]["clinical_note"]
        corpus.append(clinical_note)
    
    return corpus


# In[4]:


def top_ten_similar(dict_embeddings, target):
    def euclidean_most_similars(model, word, topn = 10):
        distances = {}
        vec1 = model[word]
        for item in model.keys():
            vec2 = model[item]
            dist = np.linalg.norm(vec1 - vec2)
            distances[item] = dist
        return(distances)
    top_10 = []
    distances = euclidean_most_similars(dict_embeddings, target)
    for word in dict_embeddings.keys():
        if word==target:
            continue
        distance = distances[word]
        if len(top_10) < 10:
            top_10.append((word, distance))
            top_10.sort(key=lambda x:x[1])
        elif distance < top_10[9][1]:
            top_10[9] = (word, distance)
            top_10.sort(key=lambda x:x[1])
    return top_10


# In[9]:


def add_domain_words(glove_dict):
    average_vector = np.zeros(glove_dict["<s>"].shape)
    count = 0
    for key in glove_dict.keys():
        count += 1
        average_vector += glove_dict[key]
    average_vector = average_vector / count

    cardiomediastinal_vector = np.zeros(glove_dict["<s>"].shape)
    for word in top_ten_similar(glove_dict, "mediastinal"):
        cardiomediastinal_vector += glove_dict[word[0]]
    cardiomediastinal_vector = cardiomediastinal_vector / 10

    ap_pa_vector = np.zeros(glove_dict["<s>"].shape)
    for word in top_ten_similar(glove_dict, "radiograph"):
        ap_pa_vector += glove_dict[word[0]]
    ap_pa_vector = ap_pa_vector / 10
    
    retrocardiac_vector = np.zeros(glove_dict["<s>"].shape)
    for word in top_ten_similar(glove_dict, "collapse"):
        retrocardiac_vector += glove_dict[word[0]]
    retrocardiac_vector = retrocardiac_vector / 10

    bibasilar_vector = np.zeros(glove_dict["<s>"].shape)
    for word in top_ten_similar(glove_dict, "lungs"):
        bibasilar_vector += glove_dict[word[0]]
    bibasilar_vector = bibasilar_vector / 10

    added_words_to_glove = ["<unk>", "cardiomediastinal", "AP", "PA", "retrocardial", "bibasilar"]
    glove_dict["<unk>"] = average_vector
    glove_dict["cardiomediastinal"] = cardiomediastinal_vector
    glove_dict["AP"] = ap_pa_vector
    glove_dict["PA"] = ap_pa_vector +0.001
    glove_dict["retrocardiac"] = retrocardiac_vector
    glove_dict["bibasilar"] = bibasilar_vector
    
    return glove_dict


# In[10]:


def main_get_word_embeddings(train_dict=None, val_dict=None, test_dict=None):
    if train_dict == None or val_dict == None or test_dict == None:
        train_dict, val_dict, test_dict = load_session_dictionary()
    corpus = build_corpus(train_dict, val_dict, test_dict)
    flattened_corpus = [item for sublist in corpus for item in sublist]
    glove_dict = main_load_glove_840B_300d("glove.840B.300d.txt", set(flattened_corpus))
    revised_glove_dict = add_domain_words(glove_dict)
    return revised_glove_dict
   


# #### Miscellaneous functions and other past approaches to embedding / Mittens model for later possible fine tuning on corpus

# In[ ]:


def write_corpus_text_file(corpus):
    with open("corpus.txt", "w") as f:
        for document in corpus:
            for word in document:
                f.write(word)
                f.write(' ')
            f.write("\n")


# In[ ]:


def top_ten_similar(dict_embeddings, target):
    def euclidean_most_similars(model, word, topn = 10):
        distances = {}
        vec1 = model[word]
        for item in model.keys():
            vec2 = model[item]
            dist = np.linalg.norm(vec1 - vec2)
            distances[item] = dist
        return(distances)
    top_10 = []
    distances = euclidean_most_similars(dict_embeddings, target)
    for word in dict_embeddings.keys():
        if word==target:
            continue
        distance = distances[word]
        if len(top_10) < 10:
            top_10.append((word, distance))
            top_10.sort(key=lambda x:x[1])
        elif distance < top_10[9][1]:
            top_10[9] = (word, distance)
            top_10.sort(key=lambda x:x[1])
    return top_10


# In[ ]:


# fixing words in corpus to lowercase if they dont appear in glove and their lowercase does
#for visit in corpus:
#    for index in range(len(visit)):
#        word = visit[index]
#        if word not in glove_dict.keys():
#            if word.lower() in glove_dict.keys():
#                visit[index] = word.lower()


# In[ ]:


def expand_glove_to_missing_words(corpus, glove, typos):
    print("-"*50)
    print("Will return word embedding dictionary")
    print("-"*50)
    final_dict = {}
    custom_glove_dict = main_load_custom_glove_model("./glove/vectors.txt")
    for word in typos:
        if word == "" or word == " ":
            continue
        custom_embedding = custom_glove_dict[word]
        final_dict[word] = custom_embedding
    for word in glove_dict.keys():
        glove_embedding = glove[word]
        final_dict[word] = glove_embedding
    print("Done!")
    print("-"*50)
    return final_dict


# In[ ]:


def make_matrix(corpus, window_size=6):
    def distinct_words(corpus):
        corpus_words = []
        num_corpus_words = -1
        flattened_corpus = [word for doc in corpus for word in doc] 
        corpus_words_set = set(flattened_corpus)
        for word in corpus_words_set: corpus_words.append(word) 
        corpus_words = sorted(corpus_words)
        num_corpus_words = len(corpus_words)
        return corpus_words, num_corpus_words
    words, num_words = distinct_words(corpus)
    M = None
    word2ind = {}
    def walk_right(string, num_steps, start, matrix, dictionary):
        count = 1
        while count <= num_steps:
            matrix[dictionary[string[start]], dictionary[string[start+count]]] += 1
            count += 1
        return matrix
    
    def walk_left(string, num_steps, start, matrix, dictionary):
        count = 1
        while count <=num_steps:
            matrix[dictionary[string[start]], dictionary[string[start-count]]] += 1
            count += 1
        return matrix
    
    M = np.zeros((num_words, num_words))
    count = 0
    for word in words:
        word2ind[word] = count
        count += 1
    for document in corpus:
        for word_index in range(len(document)):
            M = walk_right(document, num_steps=min(window_size, len(document)-word_index-1), start=word_index, matrix=M, dictionary=word2ind)
            M = walk_left(document, num_steps=min(window_size, word_index), start=word_index, matrix=M, dictionary=word2ind)
    return M, word2ind


# In[ ]:


# now we will train our embeddings on this corpus
# from mittens import Mittens
#mittens_model = Mittens(n=300, max_iter=50)
#new_embeddings = mittens_model.fit(
#    occur_matrix,
#    vocab=word2ind,
#    initial_embedding_dict= glove_all)
#fine_tuned_glove = {}
#for word in glove_all.keys():
#    try:
#        index_in_new = word2ind[word]
#        embedding_of_new = new_embeddings[index_in_new]
#        fine_tuned_glove[word] = embedding_of_new
#    except:
#        pass

