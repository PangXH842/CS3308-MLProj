import os
import random
import pickle
import argparse
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from GCN import GCN
from generate_data import generate_data
from utils import log, init_log, generate_line_graph

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
    log_file = init_log("main.py", args)
    
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
    model = GCN(args.hidden_dims)

    # Loss function: Mean Square Error
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    y_train, y_test = [], []

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
        y_train.append(avg_train_loss)
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
        y_test.append(avg_test_loss)
        log(log_file, f"Average Test Loss: {avg_test_loss:.4f} {np.round(test_loss, 4)[:20]}")

    # Save model weights
    log(log_file, f"Saving model weights to {args.save_model_path}")
    torch.save(model.state_dict(), args.save_model_path)

    # Generate line graph
    x_data = [i for i in range(1, args.epochs+1)]
    x_label = "Epochs"
    y_label = "Loss"
    title = "Training Process of GCN model"
    legend_labels = ["Train loss", "Test loss"]

    generate_line_graph(x_data, y_train, y_test, x_label, y_label, title, legend_labels, args.save_plot_path)
    
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
    parser.add_argument('--hidden_dims', type=int, default=16)
    parser.add_argument('--save_model_path', type=str, default="model_weights.pth")
    parser.add_argument('--save_plot_path', type=str, default="loss_graph.png")
    args = parser.parse_args()

    main(args)
