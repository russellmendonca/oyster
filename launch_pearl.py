import os
import json
import argparse
from setup import setup_and_run
import rlkit.torch.pytorch_util as ptu
from configs.default import default_pearl_config
from misc_utils import deep_update_dict

def main(config, log_annotation, seed):
    assert 0 <= seed <= 3

    ptu.set_gpu_mode(True)
    variant = default_pearl_config
    with open(os.path.join(config)) as f:
        exp_params = json.load(f)
    variant = deep_update_dict(exp_params, variant)

    for name, var in zip(['log_annotation', 'seed'],[log_annotation, seed]):
        if name in variant and var != None:
            variant[name] = var
    #variant['util_params']['gpu_id'] = gpu_id
    setup_and_run(variant)


parser = argparse.ArgumentParser()
parser.add_argument("config", default=None)
parser.add_argument("log_annotation", default=None)
parser.add_argument("--seed", default=None, type=int)

args = parser.parse_args()
main(args.config, args.log_annotation, args.seed)
