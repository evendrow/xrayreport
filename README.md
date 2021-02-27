# Chest X-Ray Captioning

**Requrements**
- pytorch=1.7
- torchvision
- tqdm
- pandas
- opencv
- matplotlib

### Dataset Preparation
This project uses the MIMIC-CXR dataset. The dataset can be downloaded from https://physionet.org/content/mimic-cxr/2.0.0/ after finishing the required trainings and obtaining permissions. After downloading the dataset, you can generate feature embeddings for both the single and double encoders via
` python create_dataset_script.py`
Since this requires inference over the entire dataset, this process can take a while.

### Training
After getting feature embeddings, you can train the model via
` python double_tf_training_script_vm.py`
this will save a checkpoint every epoch as well.
