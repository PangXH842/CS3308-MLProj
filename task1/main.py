import os
import torch
import numpy as np
import re
import abc_py

# Obtain current AIG
state = 'alu2_0130622'
circuitName, actions = state.split('_')
circuitPath = './InitialAIG/train/' + circuitName + '.aig'
libFile = './lib/7nm/7nm.lib'
logFile = 'alu2.log'
nextState = state + '.aig' # current AIG file
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}
action_cmd = ''
for action in actions:
    action_cmd += (synthesisOpToPosDic[int(action)]+'; ')
abcRunCmd = "./oss-cad-suite/bin/yosys-abc -c \""\
                + "read " + circuitPath + "; " + action_cmd \
                + "read_lib " + libFile + "; "\
                + "write " + nextState + "; print_stats"\
            + "\" > " + logFile
os.system(abcRunCmd)

# Evaluate AIG with yosys
abcRunCmd = "./yosys-abc -c \""\
                + "read " + circuitPath + "; "\
                + "read_lib " + libFile + "; "\
                + "map; topo; stime"\
            + "\" >" + logFile
os.system(abcRunCmd)
with open(logFile, 'r') as f:
    areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    eval = float(areaInformation[-9]) * float(areaInformation[-4])

# Regularize with resyn2
RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; "\
                + "rewrite -z; balance; refactor -z; rewrite -z; balance; "
abcRunCmd = "./yosys-abc -c \""\
                +"read " + circuitPath + "; "\
                + RESYN2_CMD \
                + "read_lib " + libFile + "; "\
                + "write" + nextState + "; "\
                + "map; topo; stime"\
            +"\" >" + logFile
                # + "write_bench -l" + nextBench + "; "\
os.system(abcRunCmd)
with open(logFile, 'r') as f:
    areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    baseline = float(areaInformation[-9]) * float(areaInformation[-4])
eval = 1 - eval/baseline

_abc = abc_py.AbcInterface ()
_abc.start()
_abc.read(state)
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
        fanin = aigNode.fanin0 ()
        edge_src_index.append(nodeIdx)
        edge_target_index.append(fanin)
    if (aigNode.hasFanin1()):
        fanin = aigNode.fanin1 ()
        edge_src_index.append(nodeIdx)
        edge_target_index.append(fanin)
data['edge_index'] = torch.tensor ([edge_src_index, edge_target_index], dtype=torch.long)
data['node_type'] = torch.tensor(data ['node_type'])
data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
data['nodes'] = numNodes