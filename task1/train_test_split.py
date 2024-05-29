import os
import shutil
import random

def split_data(source_folder, train_folder, test_folder, split_ratio=0.8):
    # Remove existing training and testing directories if they exist and create new ones
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    
    os.makedirs(train_folder)
    os.makedirs(test_folder)
    
    all_types = os.listdir("./InitialAIG/train/")
    subfolders = [x.split('.')[0] for x in all_types]
    for subfolder in subfolders:
        os.makedirs(os.path.join(train_folder, subfolder))
        os.makedirs(os.path.join(test_folder, subfolder))
    
    # Get the list of all files in the source directory
    all_files = os.listdir(source_folder)
    
    # Separate the files by their type
    for subfolder in subfolders:
        files_list = [file for file in all_files if file.startswith(subfolder)]
        random.shuffle(files_list)
        split_index = int(len(files_list) * split_ratio)
        train_files = files_list[:split_index]
        test_files = files_list[split_index:]
        for file_name in train_files:
            full_file_name = os.path.join(source_folder, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, os.path.join(train_folder, subfolder))
        for file_name in test_files:
            full_file_name = os.path.join(source_folder, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, os.path.join(test_folder, subfolder))

# Example usage
source_folder = './task1/project_data'
train_folder = './task1/train_data'
test_folder = './task1/test_data'
split_ratio = 0.8  # 80% for training, 20% for testing

split_data(source_folder, train_folder, test_folder, split_ratio)
