import os
import numpy as np
import abc_py
import torch
from torch_geometric.data import Data

from obtain_aig import obtain_aig

def build_graph(data, score):
    x = torch.tensor(list(zip(data['node_type'], data['num_inverted_predecessors'])), dtype=torch.float)
    edge_index = data["edge_index"]
    y = torch.tensor([score], dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, y=y)

    return graph

def generate_data(state, score):
    obtain_aig(state)

    temp_folder = "temp"
    state_aig = state + ".aig"
    state_aig_path = os.path.join(temp_folder, state_aig)
    _abc = abc_py.AbcInterface ()
    _abc.start()
    _abc.read(state_aig_path)
    os.remove(state_aig_path)

    data = {}
    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else :
            data['node_type'][nodeIdx] = 2
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['nodes'] = numNodes

    graph = build_graph(data, score)

    return graph

if __name__ == "__main__":
    data = generate_data('alu2_0130622', 0)
    print(data)