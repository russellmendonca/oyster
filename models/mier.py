import pickle
import os.path as osp

import numpy as np
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from models.constructor import construct_model

from .misc_utils import TensorBoardLogger


class MIER:

    def __init__(self, variant):

        for key in variant:
            setattr(self, key, variant['key'])
        self.env = NormalizedBoxEnv(ENVS[self.env_name](**self.env_params))
        self.model = construct_model(obs_dim=int(self.env.observation_space.shape[0]),
                                     act_dim=int(self.env.action_space.shape[0]),
                                     model_hyperparams=variant.model_hyperparams
                                     )

        self.logger = TensorBoardLogger(self.log_dir)
        self.loaded_data = pickle.load(open(self.data_load_path, 'rb'))['replay_buffer']
        self.loaded_data_size = [len(self.loaded_data[task]['observations']) for task in range(len(self.loaded_data))]

    def train(self):

        for epoch in range(self.num_epochs):
            self.model.save_model(osp.join(self.log_dir, 'Itr_' + str(epoch)))
            self.run_training_epoch(epoch)

    # TODO : add evaluation

    def sample_data(self, tasks):

        def get_data(task, key, idxs):
            return self.loaded_data[task][key][idxs]

        for i, task in enumerate(tasks):
            idxs = np.random.choice(np.arange(self.loaded_data_size[task]), self.batch_size)
            obs, acts, next_obs, rews = [get_data(task, key, idxs) for key in
                                         ['observations', 'actions', 'rewards', 'next_observations']]

            if i == 0:
                all_inputs = inputs;
                all_targets = targets
            else:
                all_inputs = np.concatenate([all_inputs, inputs], axis=0)
                all_targets = np.concatenate([all_targets, targets], axis=0)

        all_inputs = all_inputs * np.ones((FLAGS.num_networks,) + all_inputs.shape)
        all_targets = all_targets * np.ones((FLAGS.num_networks,) + all_targets.shape)

        return all_inputs, all_targets

    def run_training_epoch(self, epoch):

        for step in range(self.num_training_steps_per_epoch):
            # print(self.sess.run(self.meta_model.context))
            if step % 100 == 0:
                print('step ', step)

            tasks = np.random.choice(self.train_tasks, self.model.meta_train_n_tasks,
                                     replace=self.model.meta_train_n_tasks > len(self.train_tasks))

            train_input, train_target = self.sample_data(tasks)
            val_input, val_target = self.sample_data(tasks)

            feed_dict = {self.model.train_input: train_input,
                         self.model.train_target: train_target,
                         self.model.val_input: val_input,
                         self.model.val_target: val_target}

            _, updated_contexts, pre_adapt_losses, post_adapt_losses, gvs = self.sess.run(
                [self.model.metatrain_op, self.model.updated_contexts,
                 self.model.pre_adapt_losses, self.model.post_adapt_losses, self.model.gvs],
                feed_dict=feed_dict)

            if (step + 1) % FLAGS.num_training_steps_per_epoch == 0:
                self.logger.log_dict(epoch, {'Model/preAda_ModelLoss': np.mean(pre_adapt_losses),
                                             'Model/postAda_ModelLoss': np.mean(post_adapt_losses),
                                             'Model/var_norms': np.mean([np.linalg.norm(var) for _, var in gvs]),
                                             'Model/grad_norms': np.mean([np.linalg.norm(grad) for grad, _ in gvs]),
                                             'Model/updated_context_norms': np.mean(
                                                 [np.linalg.norm(context) for context in updated_contexts])
                                             })

    def _rollout_model(self, context, rollout_batch_size, **kwargs):
        print('[ Model Rollout ] Starting | Rollout length: {} | Batch size: {}'.format(
            self._rollout_length, rollout_batch_size
        ))

        if FLAGS.run_mier:
            batch = concatenate_traj_data([self.rich_experience_buffer.sample(rollout_batch_size // 20, task_id=task_id)
                                           for task_id in
                                           np.random.choice(len(self.rich_experience_buffer.buffers), 20)])
        elif FLAGS.run_mbpo:
            batch = self.replay_buffer.sample(rollout_batch_size, task_id=0)

        obs = batch['observations']
        obs_with_context = append_context_to_inputArray(obs, context) if (context is not None) else obs

        steps_added = []
        for i in range(self._rollout_length):

            action = self.sac_trainer._policy.actions_np(obs_with_context)

            next_obs, rew, term, info = self.fake_env.step(obs, action, context, **kwargs)
            steps_added.append(len(obs))

            traj_data = TrajData(obs, action, rew, next_obs, term)
            self.model_buffer.add(traj_data, task_id=0)

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print(
                    '[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break

            obs = next_obs[nonterm_mask]

        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {'mean_rollout_length': mean_rollout_length}

        return rollout_stats

    def fast_adapt_model(self):

        rb_data = self.replay_buffer.sample(FLAGS.post_adapt_batch_size, task_id=0, return_all_data=True)
        processed_data = self.prepare_data([rb_data])

        # updating the context
        for fast_step in range(200):
            _, loss = self.sess.run([self.meta_model.fast_train_op, self.meta_model.oneTask_train_loss],
                                    feed_dict={self.meta_model.pre_adapt_input: processed_data[0],
                                               self.meta_model.pre_adapt_target: processed_data[1]
                                               })
            context = self.sess.run(self.meta_model.context)

            print('loss ', loss)
            print('context ', context)

    def _regular_train_model(self, context, **kwargs):
        all_env_data = self.replay_buffer.sample(task_id=0, return_all_data=True)
        inputs, targets = self.prepare_data([all_env_data])

        inputs = np.array(inputs).squeeze()
        targets = np.array(targets).squeeze()
        inputs_with_context = append_context_to_inputArray(inputs, context)

        return self.meta_model.regular_train_simple(inputs_with_context, targets, **kwargs)

    def extrapolate(self):

        '''
        Require : Train-phase data
        Algorithm:
        1. Take multiple fast update steps on the model initialization to obtain updated context  (using true data)
        2. Take states from train phase and current data , and use this to generate synthetic data. Add synthetic data
        to the model_buffer
        3. Use synthetic data to update policy
        4. Loop
        :return:
        '''
        if FLAGS.run_mier:
            self.rich_experience_buffer = pickle.load(open(FLAGS.load_path + 'replay_buffer.pkl', 'rb'))
            self.model_buffer = MultiTaskReplayBuffer(int(1e6), 1)

            self.collect_data(task_id=0, num_steps_prior=FLAGS.num_steps_prior, sample_from_posterior=False)
            print('Fast adaptation of model')
            self.fast_adapt_model()


        elif FLAGS.run_mbpo:
            self.model_buffer = MultiTaskReplayBuffer(int(400e3), 1)

        context = self.sess.run(self.meta_model.context) if (FLAGS.context_dim > 0) else None

        for epoch in range(FLAGS.num_epochs):
            print('############## EPOCH ', epoch, ' ##################')
            self.eval_single_task(context=context, epoch=epoch, log_name='eval/avg_return')
            # if epoch == FLAGS.num_epochs - 1:
            #     pickle.dump([_ret], FLAGS.log_dir + 'return.pkl')

            if (epoch + 1) % FLAGS.sampling_freq == 0:
                # self.collect_data(task_id=0, num_steps_prior=FLAGS.num_steps_prior, sample_from_posterior=False)

                data = self.batch_sampler.sample(FLAGS.num_steps_mbpoTrain, context=context)[0]
                self.replay_buffer.add(data, 0)

                self._regular_train_model(context, batch_size=256, max_epochs=None, holdout_ratio=0.2,
                                          max_epochs_since_update=self.max_epochs_since_update)

            for step in range(FLAGS.num_steps_mbpoTrain):

                if step % 250 == 0:
                    print('step ', step)
                    if self.pred_dynamics:
                        self._rollout_model(context, rollout_batch_size=self.rollout_batch_size,
                                            deterministic=self.deterministic_model)
                    else:
                        self.add_relabelled_reward(context, rollout_batch_size=self.rollout_batch_size)

                for _ in range(2):
                    data = self.get_sac_training_batch(256, context)
                    self.sac_trainer._do_training(step, data)
