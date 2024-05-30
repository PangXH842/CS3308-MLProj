import os
import random
import pickle
import argparse
import datetime
import time
import torch
from torch_geometric.data import Data

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
    log(log_file, "main_load_train.py")

    start_time = time.time()

    # # Construct train dataset
    # log(log_file, "\nConstructing train dataset...")
    # train_data_list, train_score_list = load_data("./task1/train_data/", args.samples_per_folder, log_file)
    # train_dataset = AIGDataset(train_data_list, train_score_list)

    # # Construct test dataset
    # log(log_file, "\nConstructing test dataset...")
    # test_data_list, test_score_list = load_data("./task1/test_data/", args.samples_per_folder, log_file)
    # test_dataset = AIGDataset(test_data_list, test_score_list)

    # Load the datasets
    log(log_file, "\nLoading datasets")
    train_dataset_file = "train_dataset.pkl"
    test_dataset_file = "test_dataset.pkl"
    with open(train_dataset_file, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(test_dataset_file, 'rb') as f:
        test_dataset = pickle.load(f)

    # Build GCN model
    model = GCN()

    # Loss function: Mean Square Error
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Begin Training Loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        log(log_file, f"Epoch {epoch}/{args.epochs}")
        train_loss = 0
        for idx, (data, target) in enumerate(train_dataset):
            x = torch.tensor(list(zip(data['node_type'], data['num_inverted_predecessors'])), dtype=torch.float)
            graph_data = Data(x=x, edge_index=data['edge_index'].type(torch.long))
            target = torch.tensor([target])
            
            optimizer.zero_grad()
            out = model(graph_data)
            loss = criterion(out, target)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_dataset)
        log(log_file, f"Average Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_dataset:
                x = torch.tensor(list(zip(data['node_type'], data['num_inverted_predecessors'])), dtype=torch.float)
                graph_data = Data(x=x, edge_index=data['edge_index'].type(torch.long))
                target = torch.tensor([target])

                output = model(graph_data)
                test_loss += criterion(output, target).item()
        
        avg_test_loss = test_loss / len(test_dataset)
        log(log_file, f"Average Test Loss: {avg_test_loss:.4f}")
    
    # Print time consumed
    end_time = time.time()
    log(log_file, f"\nTime consumed: {(end_time-start_time):.2f} secs")
    return

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--samples_per_folder', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)  # Added batch_size argument
    args = parser.parse_args()

    main(args)
