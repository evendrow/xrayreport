#!/usr/bin/env python
# coding: utf-8

# #### Returns dict = {study_id: {"clinical_note": note, "images": [list of numpy images]}, next study id}. Need to pass in parameter for "train" "test" or "val" into make_dictionary

# In[2]:


import json
import numpy as np
import os
from PIL import Image


# In[16]:


def main_make_study_dictionary(train_test_val):
    with open('annotation.json') as json_file:
        data = json.load(json_file)
    dictionary = {}
    for study_session in data[train_test_val]:
        study_id = study_session["study_id"]
        if study_id in dictionary.keys():
            for image in study_session["image_path"]:
                image_dir = str(os.getcwd())+"/images/" + image
                image_dir = str(image_dir)
                if os.path.isfile(image_dir):
                    current_images = dictionary[study_id]["images"]
                    current_images.append(image)
                    current_images = list(set(current_images))
                    dictionary[study_id]["images"] = current_images
            continue
        dictionary[study_id] = {}
        list_of_words_in_report = ["<S>"]
        clinical_report = study_session["report"]
        for line in clinical_report.splitlines():
            for word in line.split():
                if "." in word:
                    if word[-1]==".":
                        list_of_words_in_report.append(word[:-1])
                        list_of_words_in_report.append(".")
                        list_of_words_in_report.append("<s>")
                    else:
                        period = word.index(".")
                        word1 = word[:period]
                        word2 = word[period+1:]
                        if word2.isnumeric():
                            list_of_words_in_report.append(word)
                        else:
                            list_of_words_in_report.append(word1)
                            list_of_words_in_report.append(".")
                            list_of_words_in_report.append("<s>")
                            list_of_words_in_report.append(word2)
                elif word[-1]=="," or word[-1]==":" or word[-1]=="–" or word[-1]==";" or word[-1]=="-":
                    list_of_words_in_report.append(word[:-1])
                    list_of_words_in_report.append(word[-1])
                elif "/" in word and word[0]!="/" and word[-1]!="/":
                    slash = word.index("/")
                    word1 = word[:slash]
                    word2 = word[slash+1:]
                    list_of_words_in_report.append(word1)
                    list_of_words_in_report.append(word2)
                else:
                    list_of_words_in_report.append(word)
        list_of_words_in_report.append("</s>")
        dictionary[study_id]["clinical_note"] = list_of_words_in_report
        dictionary[study_id]["images"] = []
        for image in study_session["image_path"]:
            image_dir = str(str(os.getcwd())+"/images/" + image)
            if os.path.isfile(image_dir):
                current_images = dictionary[study_id]["images"]
                current_images.append(image)
                current_images = list(set(current_images))
                dictionary[study_id]["images"] = current_images
    return dictionary

