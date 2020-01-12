import click
import json
import os.path as osp
from misc_utils import deep_update_dict
from configs.default import default_mier_config
from models.mier import MIER

@click.command()
@click.argument('config', default=None)
@click.argument('seed', default=0, type=int)
@click.argument('load_epoch', default=100, type=int)

def main(config, seed, load_epoch):
    variant = default_mier_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    exp_name = osp.join(variant['env_name'], 
            variant['log_annotation'], 'seed-' + str(seed))
    variant['log_dir'] = osp.join('mier_models', exp_name)
    variant['data_load_path'] = osp.join('output', exp_name, 'extra_data', 'epoch_'+str(load_epoch)+'.pkl')

    mier_obj = MIER(variant)
    mier_obj.train()


if __name__ == "__main__":
    main()
