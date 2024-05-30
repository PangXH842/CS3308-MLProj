import torch

class AIGDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, scores):
        super(AIGDataset, self).__init__()
        self.data_dict = {}
        self.data_dict["graphs"] = graphs
        self.data_dict["scores"] = scores

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        graph = self.data_dict["graphs"][idx]
        score = self.data_dict["scores"][idx]
        return graph, score
    
    def get_all(self):
        return self.data_dict
