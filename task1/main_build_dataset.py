import os
import random
import pickle
import argparse
import datetime
import torch

from GCN import GCN
from AIGDataset import AIGDataset
from evaluate_aig import evaluate_aig
from generate_data import generate_data

def log(logf, message):
    print(message)
    with open(logf, 'a') as f:
        f.write(message + "\n")
        pass

def load_data(folder_path, samples_per_folder, log_file):
    data_list, score_list = [], []
    for circuit_type in os.listdir(folder_path):
        circuit_type_path = os.path.join(folder_path, circuit_type)

        states_file_list = os.listdir(circuit_type_path)
        random.shuffle(states_file_list)
        
        for states_file in states_file_list[:samples_per_folder]:
            states_file_path = os.path.join(circuit_type_path, states_file)
            log(log_file, f"Reading file: {states_file_path}")
            with open(states_file_path, 'rb') as f:
                states = pickle.load(f)
                for state, score in zip(states['input'], states['target']):
                    data = generate_data(state)
                    data_list.append(data)
                    score_list.append(score)
    return data_list, score_list

def main(args):
    # Create log file
    log_path = "./logs/"
    os.makedirs(log_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    log_file = log_path + f"{timestamp}.log"
    with open(log_file, 'w') as f:
        pass
    log(log_file, "main_build_dataset.py")

    # Construct train dataset
    log(log_file, "\nConstructing train dataset...")
    train_data_list, train_score_list = load_data("./task1/train_data/", args.samples_per_folder, log_file)
    train_dataset = AIGDataset(train_data_list, train_score_list)

    # Construct test dataset
    log(log_file, "\nConstructing test dataset...")
    test_data_list, test_score_list = load_data("./task1/test_data/", args.samples_per_folder, log_file)
    test_dataset = AIGDataset(test_data_list, test_score_list)

    # Save the datasets into files for later loading
    log(log_file, "\nSaving datasets")
    train_dataset_file = "train_dataset.pkl"
    test_dataset_file = "test_dataset.pkl"
    with open(train_dataset_file, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(test_dataset_file, 'wb') as f:
        pickle.dump(test_dataset, f)
    return

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_per_folder', type=int, default=10)
    args = parser.parse_args()

    main(args)
