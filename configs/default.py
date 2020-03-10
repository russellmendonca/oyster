# default PEARL experiment settings
# all experiments should modify these settings only as needed
default_pearl_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5,  # dimension of the latent context vector
    net_size=300,  # number of units per FC layer in each network
    path_to_weights=None,  # path to pre-trained weights to load into networks
    env_params=dict(
        n_tasks=2,  # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True,  # shuffle the tasks after creating them
    ),
    algo_params=dict(
        exp_mode='TRAIN',  # other mode is eval
        meta_batch=16,  # number of tasks to average the gradient across
        num_iterations=500,  # number of data sampling / training iterates
        num_initial_steps=2000,  # number of transitions collected per task before training
        num_tasks_sample=5,  # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=400,  # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0,  # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=400,
        # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=2000,  # number of meta-gradient steps taken per iteration
        num_evals=2,  # number of independent evals
        num_steps_per_eval=600,  # nuumber of transitions to eval on
        batch_size=256,  # number of transitions in the RL batch
        embedding_batch_size=64,  # number of transitions in the context batch
        embedding_mini_batch_size=64,
        # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200,  # max path length for this environment
        discount=0.99,  # RL discount factor
        soft_target_tau=0.005,  # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        reward_scale=5.,
        # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False,  # whether to sparsify rewards as determined in env
        kl_lambda=.1,  # weight on KL divergence term in encoder loss
        use_information_bottleneck=True,  # False makes latent context deterministic
        use_next_obs_in_context=False,  # use next obs if it is useful in distinguishing tasks
        update_post_train=1,  # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1,  # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False,  # recurrent or permutation-invariant encoder
        dump_eval_paths=False,  # whether to save evaluation trajectories
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=False,  # debugging triggers printing and writes logs to debug directory
        docker=False,  # TODO docker is not yet supported
    )
)

default_mier_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    env_params=dict(
        n_tasks=2,  # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True,  # shuffle the tasks after creating them
    ),
    log_dir='',
    data_load_path='',
    num_epochs=500,
    batch_size=256,
    load_data_interval=10,
    num_training_steps_per_epoch=500,
    model_hyperparams=dict(
        name='BNN',
        load_saved_model_path=None,
        num_nets=1,
        num_elites=1,
        context_dim=5,
        fixed_preupdate_context=True,
        ada_state_dynamics_pred=True,
        ada_rew_pred=True,
        reward_prediction_weight=1,

        hidden_dim=200,
        meta_batch_size=10,
        fast_adapt_steps=2,
        fast_adapt_lr=0.1,
        clip_val_inner_grad=10,
        clip_val_outer_grad=10,
        reg_weight=1,
    ),
    algo_params=dict()
)
