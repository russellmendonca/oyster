import os
import argparse

from smrl.helper import get_config_str


def get_command_str(single_exp, cuda_devices, env_type, log_annotation):
    env_type = 'cheetah-vel'
    config_str = get_config_str(env_type)
    config_str += ' --log_annotation ' + str(log_annotation)
    if single_exp:
        command_str = 'python launch_main.py' + config_str
    else:
        command_str = 'parallel \'CUDA_VISIBLE_DEVICES=' + str(cuda_devices) + '\'' + \
                      ' python launch_pearl.py ' + config_str + \
                      ' --seed={1} --gpu::: 1 2 3'
    return command_str


parser = argparse.ArgumentParser()
parser.add_argument('--cuda_devices', type=str, default='0')
parser.add_argument('--env_type', type=str)
parser.add_argument('--log_annotation', type=str)
parser.add_argument('--single_exp', type=bool, default=True)
args = parser.parse_args()

os.system('export N_GPUS=4')
os.system(get_command_str(args.single_exp, args.cuda_devices, args.env_type, args.log_annotation))
