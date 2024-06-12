import torch
import argparse
from torch_geometric.loader import DataLoader

from GCN import GCN
from generate_data import generate_data
from evaluate_aig import evaluate_aig
from utils import log, init_log

def main(args):
    # Create log file
    log_file = init_log("load_model.py", args)

    # Create and load model
    log(log_file, "\nLoading model...")
    model = GCN()
    model.load_state_dict(torch.load(args.load_model_path))

    # Evaluate state
    eval_score = evaluate_aig(args.state)
    log(log_file, f"Evaluation of state '{args.state}': {eval_score}")

    # Get model prediction
    criterion = torch.nn.MSELoss()
    graph = generate_data(args.state, eval_score)
    dataloader = DataLoader([graph])
    for batch in dataloader:
        model_score = model(batch.x, batch.edge_index, batch.batch)
        log(log_file, f"Model prediction: {float(model_score)}")
        loss = criterion(model_score.squeeze(), batch.y.squeeze())
        log(log_file, f"Loss: {loss}")

    return

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str, default="alu2_0130622")
    parser.add_argument('--load_model_path', type=str, default="model_weights.pth")
    args = parser.parse_args()

    main(args)
