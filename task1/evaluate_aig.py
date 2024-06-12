import os
import re

from obtain_aig import obtain_aig

def evaluate_aig(state):
    # Obtain current aig
    obtain_aig(state)

    temp_folder = "temp"
    os.makedirs(temp_folder, exist_ok=True)

    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = circuitName + '.log'
    logFile_path = os.path.join(temp_folder, logFile)
    state_aig = state + '.aig' # current AIG file
    state_aig_path = os.path.join(temp_folder, state_aig)

    # Evaluate AIG with yosys
    abcRunCmd = "./oss-cad-suite/bin/yosys-abc -c \""\
                    + "read " + state_aig_path + "; "\
                    + "read_lib " + libFile + "; "\
                    + "map; topo; stime"\
                + "\" >" + logFile_path
    os.system(abcRunCmd)
    with open(logFile_path, 'r') as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    # print(areaInformation)
    eval = float(areaInformation[-9]) * float(areaInformation[-4])

    # Regularize with resyn2
    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; "\
                    + "rewrite -z; balance; refactor -z; rewrite -z; balance; "
    abcRunCmd = "./oss-cad-suite/bin/yosys-abc -c \""\
                    +"read " + circuitPath + "; "\
                    + RESYN2_CMD \
                    + "read_lib " + libFile + "; "\
                    + "write" + state_aig_path + "; "\
                    + "map; topo; stime"\
                +"\" >" + logFile_path
                    # + "write_bench -l" + nextBench + "; "\
    os.system(abcRunCmd)
    with open(logFile_path, 'r') as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    eval = 1 - eval/baseline
    
    os.remove(state_aig_path)

    return eval

if __name__ == "__main__":
    eval = evaluate_aig('alu2_0130622')
    print(eval)