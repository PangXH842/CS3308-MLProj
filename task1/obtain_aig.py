import os

def obtain_aig(state):
    # Obtain current AIG
    temp_folder = "temp"
    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = circuitName + '.log'
    logFile_path = os.path.join(temp_folder, logFile)
    state_aig = state + '.aig' # current AIG file
    state_aig_path = os.path.join(temp_folder, state_aig)
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
                    + "write " + state_aig_path + "; print_stats"\
                + "\" > " + logFile_path
    os.system(abcRunCmd)

if __name__ == "__main__":
    obtain_aig('alu2_0130622')