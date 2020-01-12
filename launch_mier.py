import os.path as osp

from configs.default import default_mier_config


@click.command()
@click.argument('config', default=None)
@click.argument('seed', default=None, type=int)
def main(config, seed=0):
    variant = default_mier_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    variant['log_dir'] = osp.join('mier_models', variant['env_name'],
                                  variant['log_annotation'], 'seed_' + str(seed))

    mier_obj = MIER(variant)
    mier_obj.train()


if __name__ == "__main__":
    main()
