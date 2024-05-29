import os
import random
import pickle
import argparse
import torch
import torch.utils.data as Data

from GCN import GCN
from AIGDataset import AIGDataset
from evaluate_aig import evaluate_aig
from generate_data import generate_data

def main(args):
    # Train Data Dataset
    print("\nConstructing train dataset...")
    train_folder = "./task1/train_data/"
    data_list, score_list = [], []
    for circuit_type in os.listdir(train_folder):
        circuit_type_path = os.path.join(train_folder, circuit_type)

        states_file_list = os.listdir(circuit_type_path)
        random.shuffle(states_file_list)
        
        for states_file in states_file_list[:args.samples_per_folder]:
            states_file_path = os.path.join(circuit_type_path, states_file)

            print("Reading file: " + states_file_path)
            with open(states_file_path, 'rb') as f:
                states = pickle.load(f)
                # print(states)
                for state, score in zip(states['input'], states['target']):
                    data = generate_data(state)
                    data_list.append(data)
                    score_list.append(score)
    
    train_dataset = AIGDataset(data_list, score_list)

    # Test Data Dataset
    print("\nConstructing test dataset...")
    test_folder = "./task1/test_data/"
    data_list, score_list = [], []
    for circuit_type in os.listdir(test_folder):
        circuit_type_path = os.path.join(test_folder, circuit_type)

        states_file_list = os.listdir(circuit_type_path)
        random.shuffle(states_file_list)
        
        for states_file in states_file_list[:args.samples_per_folder]:
            states_file_path = os.path.join(circuit_type_path, states_file)

            print("Reading file: " + states_file_path)
            with open(states_file_path, 'rb') as f:
                states = pickle.load(f)
                # print(states)
                for state, score in zip(states['input'], states['target']):
                    data = generate_data(state)
                    data_list.append(data)
                    score_list.append(score)
    
    test_dataset = AIGDataset(data_list, score_list)

    train_dataloader = Data.DataLoader(dataset=train_dataset)
    test_dataloader = Data.DataLoader(dataset=test_dataset)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build GCN model
    model = GCN().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Loss function: Mean Square Error
    criterion = torch.nn.MSELoss()

    # Begin Training Loop
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        for batch_idx, (graph, target) in enumerate(train_dataloader):
            graph = graph.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(graph)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for graph, target in test_dataloader:
                graph = graph.to(device)
                target = target.to(device)
                output = model(graph)
                test_loss += criterion(output, target).item()
        test_loss /= len(test_dataloader.dataset)
        print(test_loss)

    print("\nFinish!")
    return

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser() 
    parser.add_argument('--epochs', type=int, default=10) 
    parser.add_argument('--learning_rate', type=float, default=0.01)    
    parser.add_argument('--samples_per_folder', type=int, default=10)   
    args = parser.parse_args()

    main(args)
