import os
import json

import numpy as np
import click
import torch
#from learning_to_adapt.envs.ant_env import AntEnv as CrippledAntEnv

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_pearl_config
from misc_utils import deep_update_dict


def experiment(variant):
    # create multi-task environment and sample tasks
    env_name = variant['env_name']
    if env_name in ['cheetah-vel', 'cheetah-mod-control']:
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))

    elif env_name in ['ant-crippled']:
        env = NormalizedBoxEnv(CrippledAntEnv())
        # env = CrippledAntEnv()

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params'][
        'use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    target_qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    target_qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )

    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[variant['n_train_tasks']:]),
        nets=[agent, qf1, qf2, target_qf1, target_qf2],
        latent_dim=latent_dim,
        **variant['algo_params']
    )
    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))

        target_qf1.load_state_dict(torch.load(os.path.join(path, 'target_qf1.pth')))
        target_qf2.load_state_dict(torch.load(os.path.join(path, 'target_qf2.pth')))

        # TODO hacky, revisit after model refactor
        # load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))
    #import ipdb ; ipdb.set_trace()
    # optional GPU mode

    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # run the algorithm

    if variant['algo_params']['exp_mode'] == 'TRAIN':
        algorithm.train()
    elif variant['algo_params']['exp_mode'] == 'EVAL':
        assert variant['algo_params']['dump_eval_paths'] == True
        algorithm._try_to_eval()
    else:
        algorithm.eval_with_loaded_latent()


@click.command()
@click.argument('config', default=None)
@click.argument('seed', default=None, type=int)
def main(config, seed):
    assert 0 <= seed <= 3
    gpu_id = seed
    variant = default_pearl_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu_id
    exp_log_name = variant['env_name'] + '/' + variant['log_annotation'] + '/seed_' + str(seed)
    setup_logger(exp_log_name, variant=variant, exp_id=None,
                 base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode='gap',
                 snapshot_gap=10)
    # variant['path_to_weights'] = '/home/russell/oyster/output/pearl/cheetah-vel/vel-0-1/seed-0/'
    # variant['path_to_weights']  = "/home/russell/mier_proj/oyster/output/ant-crippled/regular/seed-0/"
    #variant['path_to_weights'] = "/home/russell/mier_proj/oyster/output/cheetah-mod-control/negated-joints/seed-0/itr_240/"
    #variant['algo_params']['saved_latent_dir']="/nfs/kun1/users/russell/pearl_data/cheetah-mod-control/negated-joints/seed-0/inference/"
    #variant['algo_params']['exp_mode'] = 'debug'
    
    ptu.set_gpu_mode(True)
    experiment(variant)


if __name__ == "__main__":
    main()
