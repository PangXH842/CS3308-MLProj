import os

def obtain_aig(state = 'alu2_0130622'):
    # Obtain current AIG
    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = 'alu2.log'
    state_aig = state + '.aig' # current AIG file
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
                    + "write " + state_aig + "; print_stats"\
                + "\" > " + logFile
    os.system(abcRunCmd)

if __name__ == "__main__":
    obtain_aig()