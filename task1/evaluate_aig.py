import os
import re

state = 'alu2_0130622'
circuitName, actions = state.split('_')
circuitPath = './InitialAIG/train/' + circuitName + '.aig'
libFile = './lib/7nm/7nm.lib'
logFile = 'alu2.log'

# Evaluate AIG with yosys
abcRunCmd = "./yosys-abc -c \""\
                + "read " + circuitPath + "; "\
                + "read_lib " + libFile + "; "\
                + "map; topo; stime"\
            + "\" >" + logFile
os.system(abcRunCmd)
with open(logFile) as f:
    areaInformation = re.findall ('[a-zA-Z0-9.]+', f.readlines()[-1])
    eval = float(areaInformation[-9]) * float(areaInformation[-4])