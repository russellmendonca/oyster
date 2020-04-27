import os
import numpy as np
import torch
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

def setup_and_run(variant):
    #setup env
    env_name = variant['env_name']
    env_params = variant['env_params']
    env_params['n_tasks'] = variant["n_train_tasks"] + variant["n_eval_tasks"]
    env = NormalizedBoxEnv(ENVS[env_name](**env_params))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    latent_dim = variant['latent_size']
    reward_dim = 1

    #setup encoder
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

    #setup actor, critic
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
        train_tasks=list(np.arange(variant['n_train_tasks'])),
        eval_tasks=list(np.arange([variant['n_train_tasks'],
                                   variant['n_train_tasks'] + variant['n_val_tasks']])),
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
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    if ptu.gpu_enabled():
        algorithm.to()

    #setup logger
    run_mode = variant['algo_params']['run_mode']
    exp_log_name = os.path.join(variant['env_name'], run_mode,
                                variant['log_annotation'], 'seed-' + str(variant['seed']))

    setup_logger(exp_log_name, variant=variant, exp_id=None,
                 base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode='gap',
                 snapshot_gap=10)

    # run the algorithm
    if run_mode == 'TRAIN':
        algorithm.train()
    elif run_mode == 'EVAL':
        assert variant['algo_params']['dump_eval_paths'] == True
        algorithm._try_to_eval()
    else:
        algorithm.eval_with_loaded_latent()

