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

def main(args):
    # Create log file
    log_path = "./logs/"
    os.makedirs(log_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    log_file = log_path + f"{timestamp}.log"
    with open(log_file, 'w') as f:
        pass
    log(log_file, "main_load_train.py")
    log(log_file, f"{args}")

    start_time = time.time()

    # Load the datasets
    log(log_file, "\nLoading datasets")
    train_dataset_file = "train_dataset.pkl"
    test_dataset_file = "test_dataset.pkl"
    with open(train_dataset_file, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(test_dataset_file, 'rb') as f:
        test_dataset = pickle.load(f)

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
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    main(args)
