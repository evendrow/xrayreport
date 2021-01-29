# from get_matrix_from_cnn import *
# from image_feature_dataset import ImageFeatureDataset
import os
from dataset_utils import load_mimic_data
from cnn_utils import extract_image_features, save_annotations

MIMIC_DIR = "../mimic_cxr"
EXPORT_DIR = "../mimic_features"

mimic_data = load_mimic_data(fold="val", only_one_image=True, choose_random_scan=False)


image_file_list = []
clinical_notes = []

MAX_ITER = 10
count = 0
for image, note in mimic_data.items():
	image_file_list.append(image)
	clinical_notes.append(note)

	count += 1
	if count >= MAX_ITER:
		break

print("Got", len(image_file_list), "images.")
print("Extracting image features...")

features = extract_image_features(MIMIC_DIR, image_file_list, "chexpert")

print("Saving annotations...")
save_annotations(features, clinical_notes, image_file_list, EXPORT_DIR)


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