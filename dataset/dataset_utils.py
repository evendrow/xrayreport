import json
import numpy as np
import os
import random

from PIL import Image

MIMIC_PATH = "../mimic_cxr"

def create_mimic_dictionary(fold="train"):
    '''
    Creates a clinical note dictionary from a given fold.
    - fold: One of "train", "test", or "val"

    returns dict = {
        study_id: {
            "clinical_note": note, 
            "images": [list of numpy images]
        }, ...
    }. 

    Ethan wrote this method, which is why many years from now, alien archaeologists
        will try their best, but still struggle, to decipher the meaning of this code.

    jk :D

    '''
    annotation_directory = os.path.join(MIMIC_PATH, "annotation.json")
    with open(annotation_directory) as json_file:
        data = json.load(json_file)
    dictionary = {}
    for study_session in data[fold]:
        study_id = study_session["study_id"]
        if study_id in dictionary.keys():
            for image in study_session["image_path"]:
                image_dir = os.path.join(MIMIC_PATH, "images/", image)
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
            image_dir = os.path.join(MIMIC_PATH, "images/", image)
            if os.path.isfile(image_dir):
                current_images = dictionary[study_id]["images"]
                current_images.append(image)
                current_images = list(set(current_images))
                dictionary[study_id]["images"] = current_images
    return dictionary


def load_mimic_data(fold="train", only_one_image=True, choose_random_scan=False):
    '''
        Loads mimic annotations and images for given fold
        - fold: String.
            One of "train", "test", or "val"
        - only_one_image: Boolean
            Whether to load one or all images for each report
        - choose_random_scan: Boolean
            If loading only one image, load a random image, or the first one

        returns dict = {
            study_id: {
                "clinical_note": note, 
                "images": [list of numpy images]
            }, ...
        }. 

        Ethan wrote this one as well
    ''' 

    dictionary = create_mimic_dictionary(fold=fold)
    new_dict = {}
    if only_one_image:
        for session in dictionary.keys():
            clinical_note = dictionary[session]["clinical_note"]
            all_images = dictionary[session]["images"]
            if choose_random_scan:
                num_images = len(all_images)
                choose_random = random.randint(0,num_images-1)
                image = all_images[choose_random]
                new_dict[image] = clinical_note
            else:
                image = all_images[0]
                new_dict[image] = clinical_note
    else:
        for session in dictionary.keys():
            all_images = dictionary[session]["images"]
            for image in all_images:
                if image not in new_dict.keys():
                    new_dict[image] = clinical_note
        
    return new_dict


# In[3]:


# def main_get_notes_and_image_paths(group, only_one_image=True, image_random=True):
    # return load_data(group, only_one_image, image_random)