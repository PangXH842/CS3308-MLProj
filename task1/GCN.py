import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, hidden_dims)  
        self.conv2 = GCNConv(hidden_dims, hidden_dims)  
        self.fc = torch.nn.Linear(hidden_dims, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Global mean pooling
        x = self.fc(x)
        
        return x
