import os
import re

def evaluate_aig(state = 'alu2_0130622'):
    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = 'alu2.log'
    state_aig = state + '.aig' # current AIG file

    # Evaluate AIG with yosys
    abcRunCmd = "./oss-cad-suite/bin/yosys-abc -c \""\
                    + "read " + state_aig + "; "\
                    + "read_lib " + libFile + "; "\
                    + "map; topo; stime"\
                + "\" >" + logFile
    os.system(abcRunCmd)
    with open(logFile, 'r') as f:
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
                    + "write" + state_aig + "; "\
                    + "map; topo; stime"\
                +"\" >" + logFile
                    # + "write_bench -l" + nextBench + "; "\
    os.system(abcRunCmd)
    with open(logFile, 'r') as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    eval = 1 - eval/baseline

    return eval

if __name__ == "__main__":
    eval = evaluate_aig()
    print(eval)