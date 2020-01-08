
import os

import os
import argparse
import absl.app
from smrl.mier import MIER
from smrl.helper import get_config_str
import argparse
import json

CUDA_DEVICES='0 1 2 3'
def get_command_str(single_seed, log_annotation, pearl_config, model_config):
    config_str = get_config_str(env_type)
    config_str+= ' --log_annotation '+str(log_annotation)
    if single_exp:
        command_str = 'python run_pearl.py & python run_meta_model.py'
    else:
        command_str  = 'parallel \'CUDA_VISIBLE_DEVICES=' + str(CUDA_DEVICES)+'\''+\
    ' python launch_main.py ' + config_str+\
    ' --seed={1} ::: 1 2 3'
    return command_str

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_devices', type = str, default = '0')
parser.add_argument('--env_type' , type = str)
parser.add_argument('--log_annotation', type = str)
parser.add_argument('--single_exp', type=bool, default=True)
args = parser.parse_args()

os.system('export N_GPUS=4')
os.system(get_command_str(args.single_exp, args.cuda_devices, args.env_type, args.log_annotation ))


def main():
    os.system('python process1.py & python process2.py')
   
if __name__ == "__main__":
    main()

