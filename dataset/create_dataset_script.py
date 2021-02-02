# from get_matrix_from_cnn import *
# from image_feature_dataset import ImageFeatureDataset
import os
import json
from dataset_utils import load_mimic_data
from cnn_utils import get_cnn, extract_image_features, save_annotations, save_annotations_double, save_feature_csv
from tqdm import tqdm

MIMIC_DIR = "../mimic_cxr"
EXPORT_DIR = "../mimic_features_double"
BATCH_SIZE=64

with open('word2ind.json') as json_file: 
    word2ind = json.load(json_file)
    unk_token = word2ind["<unk>"]

def tokenize_caption(caption):
    return [(word2ind[x] if x in word2ind else unk_token) for x in caption]

def create_dataset(fold="val", max_iter=32, features=["densenet121"]):
    print("Loading mimic data...")
    mimic_data = load_mimic_data(fold=fold, only_one_image=False, choose_random_scan=True)


    image_file_list = []
    clinical_notes = []

    count = 0
    for image, note in mimic_data.items():
        image_file_list.append(image)
        clinical_notes.append(tokenize_caption(note))

        count += 1
        if count >= max_iter:
            break

    print("Got", len(image_file_list), "images.")
    print("Extracting and saving image features...")


    save_path = os.path.join(EXPORT_DIR, fold)

    model_dict = { model_name: get_cnn(model_name) for model_name in features } 

    for i in tqdm(range(0, len(image_file_list), BATCH_SIZE)):
        images = image_file_list[i:min(i+BATCH_SIZE,len(image_file_list))]
        
        features_list = []
        for f in features:
            features_list.append(extract_image_features(MIMIC_DIR, images, model_dict[f][0], model_dict[f][1]))

        if len(features) == 1:
            save_annotations(features_list[0], clinical_notes, images, save_path)
        else:
            save_annotations_double(features_list[0], features_list[1], clinical_notes, images, save_path)


    save_feature_csv(image_file_list, save_path)



if __name__ == "__main__":
    create_dataset(fold="train", max_iter=128, features=["chexpert", "densenet121"])
    create_dataset(fold="val", max_iter=128, features=["chexpert", "densenet121"])



# EXPORT_DIR = "../../mimic_cxr"
# image_list = [
#     "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
#     "p10/p10000032/s50414267/174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg",
#     "p10/p10000032/s53189527/2a2277a9-b0ded155-c0de8eb9-c124d10e-82c5caab.jpg",
#     "p10/p10000032/s53189527/e084de3b-be89b11e-20fe3f9f-9c8d8dfe-4cfd202c.jpg"
# ]
# model_name = "chexpert"
# save_path = "../mimic_feats"

# feat = extract_image_features(data_dir, image_list, model_name)
# save_feature_matrix(feat, image_list, save_path)
