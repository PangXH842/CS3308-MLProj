import os
import random
import pickle
import argparse
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from tqdm import tqdm  # for progress tracking

from GCN import GCN
from AIGDataset import AIGDataset
from evaluate_aig import evaluate_aig
from generate_data import generate_data

def load_data(folder_path, samples_per_folder):
    data_list, score_list = [], []
    for circuit_type in os.listdir(folder_path):
        circuit_type_path = os.path.join(folder_path, circuit_type)

        states_file_list = os.listdir(circuit_type_path)
        random.shuffle(states_file_list)
        
        for states_file in states_file_list[:samples_per_folder]:
            states_file_path = os.path.join(circuit_type_path, states_file)
            print(f"Reading file: {states_file_path}")
            with open(states_file_path, 'rb') as f:
                states = pickle.load(f)
                for state, score in zip(states['input'], states['target']):
                    data = generate_data(state)
                    data_list.append(data)
                    score_list.append(score)
    return data_list, score_list

def main(args):
    # Construct train dataset
    print("\nConstructing train dataset...")
    train_data_list, train_score_list = load_data("./task1/train_data/", args.samples_per_folder)
    train_dataset = AIGDataset(train_data_list, train_score_list)

    # Construct test dataset
    print("\nConstructing test dataset...")
    test_data_list, test_score_list = load_data("./task1/test_data/", args.samples_per_folder)
    test_dataset = AIGDataset(test_data_list, test_score_list)

    # DataLoaders
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build GCN model
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Loss function: Mean Square Error
    criterion = torch.nn.MSELoss()

    # Begin Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (graph, target) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}")):
            graph = graph.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(graph)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for graph, target in test_dataloader:
                graph = graph.to(device)
                target = target.to(device)
                output = model(graph)
                loss = criterion(output, target)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_dataloader)
        print(f"Average Test Loss: {avg_test_loss:.4f}")

    print("\nFinish!")
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
