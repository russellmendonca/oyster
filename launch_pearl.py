import os
import json
import argparse
from setup import setup_and_run
import rlkit.torch.pytorch_util as ptu
from configs.default import default_pearl_config
from misc_utils import deep_update_dict

def main(args):
    
    variant = default_pearl_config
    with open(os.path.join(args.config)) as f:
        exp_params = json.load(f)
    variant = deep_update_dict(exp_params, variant)

    variant_name = ''
    for name in [var_arg[0] for var_arg in get_variant_args()]:
        var_value = getattr(args, name)       
        if var_value != None:
            if name not in ['seed', 'task_id_for_extrapolation', 'log_annotation']:
                variant_name+= '_'+name+'_'+str(var_value)
            if name in variant:
                variant[name] = getattr(args, name)
            elif name in variant['algo_params']:
                variant['algo_params'][name] = var_value
          
    variant['variant_name'] = variant_name
    setup_and_run(variant)


def get_variant_args():
    return [('log_annotation',), ('num_train_steps_per_itr', int) , ('seed', int)]

parser = argparse.ArgumentParser()
parser.add_argument("config", default=None)

for variant_arg in get_variant_args():
    if len(variant_arg) == 1:
        parser.add_argument("--"+variant_arg[0], default=None)
    else:
        parser.add_argument("--"+variant_arg[0], default=None, type=variant_arg[1])

args = parser.parse_args()
main(args)

