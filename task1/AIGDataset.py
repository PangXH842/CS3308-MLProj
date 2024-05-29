import torch

class AIGDataset(torch.utils.data.Dataset):
    def __init__ (self, graphs, scores):
        super(AIGDataset, self).__init__()
        self.data_dict = {'graphs_list': graphs, 'scores_list': scores}

    def __len__(self):
        return len(self.data_dict['graphs_list'])

    def __getitem__(self, idx):
        graph = self.data_dict['graphs_list'][idx]
        score = self.data_dict['scores_list'][idx]
        return score, graph
    
    def get_data(self):
        return self.data_dict
