#!/usr/bin/env python
# coding: utf-8

# In[1]:


from make_study_dictionary import *
import random


# In[2]:


def load_data(group, only_one_image, image_random):

    dictionary = main_make_study_dictionary(group)
    new_dict = {}
    if only_one_image:
        for session in dictionary.keys():
            clinical_note = dictionary[session]["clinical_note"]
            all_images = dictionary[session]["images"]
            if image_random == False:
                image = all_images[0]
                new_dict[image] = clinical_note
            else:   
                num_images = len(all_images)
                choose_random = random.randint(0,num_images-1)
                image = all_images[choose_random]
                new_dict[image] = clinical_note
    else:
        for session in dictionary.keys():
            all_images = dictionary[session]["images"]
            for image in all_images:
                if image not in new_dict.keys():
                    new_dict[image] = clinical_note
        
    return new_dict


# In[3]:


def main_get_notes_and_image_paths(group, only_one_image=True, image_random=True):
    return load_data(group, only_one_image, image_random)


# In[ ]:




