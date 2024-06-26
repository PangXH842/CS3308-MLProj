import os
import random
import pickle
import argparse
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from generate_data import generate_data
from utils import log, init_log

def load_data(folder_path, samples_per_folder, log_file):
    states_list = []
    graphs = []
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
                    if state in states_list:
                        continue
                    states_list.append(state)
                    graph = generate_data(state, score)
                    graphs.append(graph)
    return graphs

def main(args):
    # Create log file
    log_file = init_log("main_build_dataset.py", args)
    
    start_time = time.time()

    # Construct train dataset
    log(log_file, "\nConstructing train dataset...")
    train_dataset = load_data("./task1/train_data/", args.samples_per_folder, log_file)

    train_dataset_file = "train_dataset.pkl"
    with open(train_dataset_file, 'wb') as f:
        pickle.dump(train_dataset, f)

    # Construct test dataset
    log(log_file, "\nConstructing test dataset...")
    test_dataset = load_data("./task1/test_data/", args.samples_per_folder, log_file)

    test_dataset_file = "test_dataset.pkl"
    with open(test_dataset_file, 'wb') as f:
        pickle.dump(test_dataset, f)

    # Print time consumed
    end_time = time.time()
    log(log_file, f"\nTime consumed: {(end_time-start_time):.2f} secs")
    return

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_per_folder', type=int, default=5)
    args = parser.parse_args()

    main(args)
