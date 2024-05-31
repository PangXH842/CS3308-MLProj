import os
import random
import pickle
import argparse
import datetime
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from GCN import GCN
from evaluate_aig import evaluate_aig
from generate_data import generate_data

def log(logf, message):
    print(message)
    with open(logf, 'a') as f:
        f.write(message + "\n")

def load_data(folder_path, samples_per_folder, log_file):
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
                    graph = generate_data(state, score)
                    graphs.append(graph)
    return graphs

def main(args):
    # Create log file
    log_path = "./logs/"
    os.makedirs(log_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    log_file = log_path + f"{timestamp}.log"
    with open(log_file, 'w') as f:
        pass
    log(log_file, "main.py")
    log(log_file, f"{args}")
    
    start_time = time.time()

    # Construct train dataset
    log(log_file, "\nConstructing train dataset...")
    train_dataset = load_data("./task1/train_data/", args.samples_per_folder, log_file)

    # Construct test dataset
    log(log_file, "\nConstructing test dataset...")
    test_dataset = load_data("./task1/test_data/", args.samples_per_folder, log_file)

    # Data loader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Build GCN model
    model = GCN()

    # Loss function: Mean Square Error
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Begin Training Loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        log(log_file, f"Epoch {epoch}/{args.epochs}")
        train_loss = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
        
        avg_train_loss = np.mean(train_loss)
        log(log_file, f"Average Train Loss: {avg_train_loss:.4f} {np.round(train_loss, 4)[:20]}")

        # Validation
        model.eval()
        test_loss = []
        with torch.no_grad():
            for batch in test_dataloader:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y)
                test_loss.append(loss.item())
                
        avg_test_loss = np.mean(test_loss)
        log(log_file, f"Average Test Loss: {avg_test_loss:.4f} {np.round(test_loss, 4)[:20]}")
    
    # Print time consumed
    end_time = time.time()
    log(log_file, f"\nTime consumed: {(end_time-start_time):.2f} secs")
    return

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--samples_per_folder', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    main(args)
