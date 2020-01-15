from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pickle
import os
import time
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from models.utils import TensorStandardScaler
from models.fc import FC
from utils.logging import Progress, Silent

np.set_printoptions(precision=5)


class BNN:
    """Neural network models which model aleatoric uncertainty (and possibly epistemic uncertainty
    with ensembling).
    """

    def __init__(self, sess, obs_dim, act_dim, model_hyperparams):
        """Initializes a class instance.

        """
        self._sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.output_dim = obs_dim + 1
        for key in model_hyperparams:
            setattr(self, key, model_hyperparams[key])

        # Instance variables
        self.finalized = False
        self.layers, self.max_logvar, self.min_logvar = [], None, None
        self.decays, self.optvars, self.nonoptvars = [], [], []
        self.end_act, self.end_act_name = None, None
        self.scaler = None

        # Training objects
        self.sy_train_in, self.sy_train_targ = None, None
        self.train_op, self.mse_loss = None, None

        # Prediction objects
        self.sy_pred_in2d, self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = None, None, None
        self.sy_pred_mean2d, self.sy_pred_var2d = None, None
        self.sy_pred_in3d, self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac = None, None, None

        if self.num_nets == 1:
            print("Created a neural network with variance predictions.")
        else:
            print(
                "Created an ensemble of {} neural networks with variance predictions | Elites: {}".format(self.num_nets,
                                                                                                          self.num_elites))

    @property
    def is_probabilistic(self):
        return True

    @property
    def is_tf_model(self):
        return True

    @property
    def sess(self):
        return self._sess

    ###################################
    # Network Structure Setup Methods #
    ###################################

    def add(self, layer):
        """Adds a new layer to the network.

        Arguments:
            layer: (layer) The new layer to be added to the network.
                   If this is the first layer, the input dimension of the layer must be set.

        Returns: None.
        """
        if self.finalized:
            raise RuntimeError("Cannot modify network structure after finalizing.")
        if len(self.layers) == 0 and layer.get_input_dim() is None:
            raise ValueError("Must set input dimension for the first layer.")
        #if self.model_loaded:
        #    raise RuntimeError("Cannot add layers to a loaded model.")

        layer.set_ensemble_size(self.num_nets)
        if len(self.layers) > 0:
            layer.set_input_dim(self.layers[-1].get_output_dim())
        self.layers.append(layer.copy())

    def pop(self):
        """Removes and returns the most recently added layer to the network.

        Returns: (layer) The removed layer.
        """
        if len(self.layers) == 0:
            raise RuntimeError("Network is empty.")
        if self.finalized:
            raise RuntimeError("Cannot modify network structure after finalizing.")
        #if self.model_loaded:
        #    raise RuntimeError("Cannot remove layers from a loaded model.")

        return self.layers.pop()

    def finalize(self, optimizer, optimizer_args=None, *args, **kwargs):
        """Finalizes the network.

        Arguments:
            optimizer: (tf.train.Optimizer) An optimizer class from those available at tf.train.Optimizer.
            optimizer_args: (dict) A dictionary of arguments for the __init__ method of the chosen optimizer.

        Returns: None
        """
        if len(self.layers) == 0:
            raise RuntimeError("Cannot finalize an empty network.")
        if self.finalized:
            raise RuntimeError("Can only finalize a network once.")

        self.meta_optimizer = tf.train.AdamOptimizer(1e-3)
        #self.fast_optimizer = tf.train.AdamOptimizer(1e-1)
        #self.regular_optimizer = tf.train.AdamOptimizer(1e-3)

        # Add variance output.
        self.layers[-1].set_output_dim(2 * self.layers[-1].get_output_dim())

        # Remove last activation to isolate variance from activation function.
        self.end_act = self.layers[-1].get_activation()
        self.end_act_name = self.layers[-1].get_activation(as_func=False)
        self.layers[-1].unset_activation()

        # Construct all variables.
        with self.sess.as_default():
            with tf.variable_scope(self.name):
                self.scaler = TensorStandardScaler(self.layers[0].get_input_dim())
                self.max_state_logvar = tf.Variable(np.ones([1, self.obs_dim]) * 0.5,
                                                    dtype=tf.float32, name="max_state_log_var")
                self.min_state_logvar = tf.Variable(-np.ones([1, self.obs_dim]) * 10,
                                                    dtype=tf.float32, name="min_state_log_var")

                self.max_rew_logvar = tf.Variable(
                    np.ones([1, 1]) * (0.5 + np.log(1e-10 + self.reward_prediction_weight ** 2)),
                    dtype=tf.float32,
                    name="max_rew_log_var")
                self.min_rew_logvar = tf.Variable(
                    np.ones([1, 1]) * (-10 + np.log(1e-10 + self.reward_prediction_weight ** 2)),
                    dtype=tf.float32,
                    name="min_rew_log_var")

                self.context = tf.Variable(np.zeros(self.context_dim), dtype=tf.float32, name='context')

                for i, layer in enumerate(self.layers):
                    with tf.variable_scope("Layer%i" % i):
                        layer.construct_vars()
                        self.decays.extend(layer.get_decays())
                        self.optvars.extend(layer.get_vars())

        self.nonoptvars.extend(self.scaler.get_vars())
        self.optvars.extend([self.max_state_logvar, self.min_state_logvar, self.max_rew_logvar, self.min_rew_logvar])

        if self.fixed_preupdate_context:
            self.nonoptvars.append(self.context)
        else:
            self.optvars.append(self.context)

        # Set up training
        with tf.variable_scope(self.name):
            # size is number of tasks X num nets X batch size X data_dim
            self.train_input = tf.placeholder(dtype=tf.float32,
                                              shape=[self.num_nets, None,
                                                     self.layers[0].get_input_dim() - self.context_dim],
                                              name="train_input")
            self.train_target = tf.placeholder(dtype=tf.float32,
                                               shape=[self.num_nets, None, self.layers[-1].get_output_dim() // 2],
                                               name="train_target")

            self.val_input = tf.placeholder(dtype=tf.float32,
                                            shape=[self.num_nets, None,
                                                   self.layers[0].get_input_dim() - self.context_dim],
                                            name="val_input")
            self.val_target = tf.placeholder(dtype=tf.float32,
                                             shape=[self.num_nets, None, self.layers[-1].get_output_dim() // 2],
                                             name="val_target")

            self.task_invariant_pred_loss = self._get_model_loss_dict(self.train_input, self.train_target,
                                                   self.tile_context(self.context, tf.shape(self.train_input)[1]),
                                                   add_rew_pred=not(self.ada_rew_pred),
                                                   add_state_pred=not(self.ada_state_dynamics_pred))['total_model_loss']

            self.meta_train_dict = self.compile_adaptation_step(num_tasks = self.meta_batch_size)
            self.eval_dict       = self.compile_adaptation_step(num_tasks = self.num_eval_tasks)
            
            self.total_loss = self.meta_train_dict['post_adapt_val_dict']['total_model_loss'] + self.task_invariant_pred_loss
                              #(self.reg_weight / self.meta_batch_size) * tf.norm(self.tiled_updated_contexts - self.context) + \
                              #self.task_invariant_pred_loss
                #(self.task_invariant_pred_loss*0)
                              #(self.ada_rew_pred == False) * self.train_dicts[0]['total_rew_loss'] + \
                              #(self.ada_state_dynamics_pred == False) * self.train_dicts[0]['total_state_loss']
        
            gvs = self.meta_optimizer.compute_gradients(loss=self.total_loss, var_list=self.optvars)
            self.gvs = [(tf.clip_by_value(grad, -self.clip_val_outer_grad, self.clip_val_outer_grad), var) for grad, var
                        in gvs]
            self.metatrain_op = self.meta_optimizer.apply_gradients(gvs)

            ######### extrapolation op ################
            # self.oneTask_train_loss = self.train_loss(self.train_input[0], self.train_target[0], self.context)
            #self.fast_train_op = self.fast_optimizer.minimize(self.pre_adapt_losses, var_list=[self.context])

            ########################## Regular training ops and setup ############################

#            with tf.variable_scope(self.name + '_regular'):
#                self.sy_train_in = tf.placeholder(dtype=tf.float32,
#                                                  shape=[self.num_nets, None, self.layers[0].get_input_dim()],
#                                                  name="training_inputs")
#                self.sy_train_targ = self.train_target
#
#                train_loss = tf.reduce_sum(
#                    self._compile_losses(self.sy_train_in, self.sy_train_targ, inc_var_loss=True))
#                train_loss += tf.add_n(self.decays)
#                self.regular_train_loss = train_loss + 0.01 * tf.reduce_sum(self.max_logvar) - 0.01 * tf.reduce_sum(
#                    self.min_logvar)
#                self.mse_loss = self._compile_losses(self.sy_train_in, self.sy_train_targ, inc_var_loss=False)
#
#                reg_opt_vars = self.optvars.copy()
#                if self.context in self.optvars:
#                    reg_opt_vars.remove(self.context)
#                self.regular_train_op = self.regular_optimizer.minimize(self.regular_train_loss, var_list=reg_opt_vars)

            #########################################################################################

        # Initialize all variables
        self.sess.run(tf.variables_initializer(self.optvars + self.nonoptvars +
                                               self.meta_optimizer.variables())) 
                                               #+ self.fast_optimizer.variables() +
                                               #self.regular_optimizer.variables()))

        ###### Setup prediction variables ######################
        with tf.variable_scope(self.name + '_regular'):
            self.sy_pred_in2d = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.layers[0].get_input_dim()],
                                               name="2D_training_inputs")
            self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = \
                self.create_prediction_tensors(self.sy_pred_in2d, factored=True)
            self.sy_pred_mean2d = tf.reduce_mean(self.sy_pred_mean2d_fac, axis=0)
            self.sy_pred_var2d = tf.reduce_mean(self.sy_pred_var2d_fac, axis=0) + \
                                 tf.reduce_mean(tf.square(self.sy_pred_mean2d_fac - self.sy_pred_mean2d), axis=0)

            self.sy_pred_in3d = tf.placeholder(dtype=tf.float32,
                                               shape=[self.num_nets, None, self.layers[0].get_input_dim()],
                                               name="3D_training_inputs")
            self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac = \
                self.create_prediction_tensors(self.sy_pred_in3d, factored=True)

        ######################################################################################
        self.finalized = True
    
    def compile_adaptation_step(self, num_tasks):
        updated_context_list, tiled_updated_contexts, \
            train_dicts = self.compile_updated_contexts(self.train_input, self.train_target, num_tasks)

        post_adapt_val_dict = self._get_model_loss_dict(self.val_input, self.val_target,
                                                                 tiled_updated_contexts,
                                                                 add_rew_pred=self.ada_rew_pred,
                                                                 add_state_pred=self.ada_state_dynamics_pred)

        return {'post_adapt_val_dict': post_adapt_val_dict, 
                'train_dicts': train_dicts, 
                'updated_context_list': updated_context_list}

    def tile_context(self, context, input_dim):
        return tf.tile(context[None, None, :], (self.num_nets, input_dim ,1))

    def process_contexts(self, context_list, tiled_grads, num_tasks, num_data_points):
        task_grads = [tf.reduce_mean(tf.reshape(task_grad, (-1, self.context_dim)), axis=0) for task_grad in
                      tf.split(tiled_grads, num_tasks, axis=1)]
        # updated_contexts = tiled_contexts - self.fast_adapt_lr * grad
        updated_context_list = [ context - self.fast_adapt_lr * grad for context, grad in zip(context_list, task_grads)]
        tiled_updated_contexts =  tf.concat([self.tile_context(context, num_data_points//num_tasks)
                                                                   for context in updated_context_list], axis = 1)
    
        return updated_context_list , tiled_updated_contexts

    def compile_updated_contexts(self, train_input, train_target, num_tasks):

        all_dicts = []
        assert self.num_nets == 1
        num_data_points = tf.shape(train_input)[1]
        tiled_contexts = self.tile_context(self.context, num_data_points)
        pre_adapt_dict = self._get_model_loss_dict(train_input, train_target,
                                                   tiled_contexts,
                                                   add_rew_pred=self.ada_rew_pred,
                                                   add_state_pred=self.ada_state_dynamics_pred)

        grads = tf.clip_by_value(tf.gradients(pre_adapt_dict['total_model_loss'], tiled_contexts)[0],
                                -self.clip_val_inner_grad, self.clip_val_inner_grad)
        
        pre_adapt_dict['grad_norm'] = tf.norm(grads)
        all_dicts.append(pre_adapt_dict)

        updated_context_list, tiled_updated_contexts = self.process_contexts([self.context for _ in range(num_tasks)], grads, num_tasks, num_data_points)

        for step in range(self.fast_adapt_steps):
            post_adapt_dict = self._get_model_loss_dict(train_input, train_target,
                                                        tiled_updated_contexts,
                                                        add_rew_pred=self.ada_rew_pred,
                                                        add_state_pred=self.ada_state_dynamics_pred)

            grads = tf.clip_by_value(tf.gradients(post_adapt_dict['total_model_loss'], tiled_updated_contexts)[0],
                                    -self.clip_val_inner_grad, self.clip_val_inner_grad)
            post_adapt_dict['grad_norm'] = tf.norm(grads)
            all_dicts.append(post_adapt_dict)
            if step < self.fast_adapt_steps-1:
                updated_context_list, tiled_updated_contexts = self.process_contexts(updated_context_list, grads, num_tasks, num_data_points)
     
        return updated_context_list, tiled_updated_contexts, all_dicts

    def _save_state(self, idx):
        self._state[idx] = [layer.get_model_vars(idx, self.sess) for layer in self.layers]

    def _set_state(self):
        keys = ['weights', 'biases']
        ops = []
        num_layers = len(self.layers)
        for layer in range(num_layers):
            # net_state = self._state[i]
            params = {key: np.stack([self._state[net][layer][key] for net in range(self.num_nets)]) for key in keys}
            ops.extend(self.layers[layer].set_model_vars(params))
        self.sess.run(ops)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                self._save_state(i)
                updated = True
                improvement = (best - current) / best
                # print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(epoch, i, improvement, best, current))
        print('################################')
        print(self._epochs_since_update)
        print('###############################')

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            # print('[ BNN ] Breaking at epoch {}: {} epochs since update ({} max)'.format(epoch, self._epochs_since_update, self._max_epochs_since_update))
            return True
        else:
            return False

    def _start_train(self):
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.num_nets)}
        self._epochs_since_update = 0

    def _end_train(self, holdout_losses):
        sorted_inds = np.argsort(holdout_losses)
        self._model_inds = sorted_inds[:self.num_elites].tolist()
        print('Using {} / {} models: {}'.format(self.num_elites, self.num_nets, self._model_inds))

    def random_inds(self, batch_size):
        inds = np.random.choice(self._model_inds, size=batch_size)
        return inds

    def reset(self):
        print('[ BNN ] Resetting model')
        [layer.reset(self.sess) for layer in self.layers]

    def get_val_loss(self, inputs, targets):

        holdout_losses = self.sess.run(
            [self.mse_loss, self.regular_train_loss],
            feed_dict={
                self.sy_train_in: inputs,
                self.sy_train_targ: targets
            }
        )
        return holdout_losses

    #################
    # Model Methods #
    #################
    def regular_train_simple(self, inputs, targets,
                             batch_size=32, max_epochs=5, holdout_ratio=0.2):
        """Trains/Continues network training

        Arguments:
            inputs (np.ndarray): Network inputs in the training dataset in rows.
            targets (np.ndarray): Network target outputs in the training dataset in rows corresponding
                to the rows in inputs.
            batch_size (int): The minibatch size to be used for training.
            epochs (int): Number of epochs (full network passes that will be done.
            hide_progress (bool): If True, hides the progress bar shown at the beginning of training.

        Returns: None
        """

        model_metrics = {}

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        # Split into training and holdout sets
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])

        model_metrics['initial_val_mse_loss'] = self.get_val_loss(holdout_inputs, holdout_targets)

        print('[ BNN ] Training {} | Holdout: {}'.format(inputs.shape, holdout_inputs.shape))
        with self.sess.as_default():
            self.scaler.fit(inputs)

        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])

        for epoch in range(max_epochs):
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                self.sess.run(
                    self.regular_train_op,
                    feed_dict={self.sy_train_in: inputs[batch_idxs], self.sy_train_targ: targets[batch_idxs]}
                )

            idxs = shuffle_rows(idxs)

        model_metrics['final_val_mse_loss'] = self.get_val_loss(holdout_inputs, holdout_targets)
        print(model_metrics)
        return OrderedDict(model_metrics)

    def regular_train(self, inputs, targets,
                      batch_size=32, max_epochs=None, max_epochs_since_update=5,
                      hide_progress=False, holdout_ratio=0.0, max_logging=5000, max_grad_updates=None, timer=None,
                      max_t=None):
        """Trains/Continues network training

        Arguments:
            inputs (np.ndarray): Network inputs in the training dataset in rows.
            targets (np.ndarray): Network target outputs in the training dataset in rows corresponding
                to the rows in inputs.
            batch_size (int): The minibatch size to be used for training.
            epochs (int): Number of epochs (full network passes that will be done.
            hide_progress (bool): If True, hides the progress bar shown at the beginning of training.

        Returns: None
        """
        self._max_epochs_since_update = max_epochs_since_update
        self._start_train()
        model_metrics = {}
        break_train = False

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        # Split into training and holdout sets
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])

        print('[ BNN ] Training {} | Holdout: {}'.format(inputs.shape, holdout_inputs.shape))
        with self.sess.as_default():
            self.scaler.fit(inputs)

        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])
        if hide_progress:
            progress = Silent()
        else:
            progress = Progress(max_epochs)

        if max_epochs:
            epoch_iter = range(max_epochs)
        else:
            epoch_iter = itertools.count()

        model_metrics['initial_val_mse_loss'] = self.get_val_loss(holdout_inputs, holdout_targets)
        # else:
        #     epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
        all_contexts = []
        t0 = time.time()
        grad_updates = 0
        for epoch in epoch_iter:
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                self.sess.run(
                    self.train_op,
                    feed_dict={self.sy_train_in: inputs[batch_idxs], self.sy_train_targ: targets[batch_idxs]}
                )
                all_contexts.append(self.sess.run(self.context))
                grad_updates += 1

            idxs = shuffle_rows(idxs)
            if not hide_progress:
                if holdout_ratio < 1e-12:
                    losses = self.sess.run(
                        self.mse_loss,
                        feed_dict={
                            self.sy_train_in: inputs[idxs[:, :max_logging]],
                            self.sy_train_targ: targets[idxs[:, :max_logging]]
                        }
                    )
                    named_losses = [['M{}'.format(i), losses[i]] for i in range(len(losses))]
                    progress.set_description(named_losses)
                else:
                    losses = self.sess.run(
                        self.mse_loss,
                        feed_dict={
                            self.sy_train_in: inputs[idxs[:, :max_logging]],
                            self.sy_train_targ: targets[idxs[:, :max_logging]]
                        }
                    )
                    holdout_losses = self.sess.run(
                        self.mse_loss,
                        feed_dict={
                            self.sy_train_in: holdout_inputs,
                            self.sy_train_targ: holdout_targets
                        }
                    )
                    named_losses = [['M{}'.format(i), losses[i]] for i in range(len(losses))]
                    named_holdout_losses = [['V{}'.format(i), holdout_losses[i]] for i in range(len(holdout_losses))]
                    named_losses = named_losses + named_holdout_losses + [['T', time.time() - t0]]
                    progress.set_description(named_losses)
                    break_train = self._save_best(epoch, holdout_losses)

            progress.update()
            t = time.time() - t0
            if break_train or (max_grad_updates and grad_updates > max_grad_updates):
                break
            if max_t and t > max_t:
                descr = 'Breaking because of timeout: {}! (max: {})'.format(t, max_t)
                progress.append_description(descr)
                # print('Breaking because of timeout: {}! | (max: {})\n'.format(t, max_t))
                # time.sleep(5)
                break

        progress.stamp()
        if timer: timer.stamp('bnn_train')

        self._set_state()
        if timer: timer.stamp('bnn_set_state')

        model_metrics['final_val_mse_loss'] = self.get_val_loss(holdout_inputs, holdout_targets)
        print(model_metrics)
        return OrderedDict(model_metrics)

    def predict(self, inputs, factored=False, *args, **kwargs):
        """Returns the distribution predicted by the model for each input vector in inputs.
        Behavior is affected by the dimensionality of inputs and factored as follows:

        inputs is 2D, factored=True: Each row is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of shape
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector j.

        inputs is 2D, factored=False: Each row is treated as an input vector.
            Returns a mean of shape [batch_size, output_dim] and variance of shape
            [batch_size, output_dim], where aggregation is performed as described in the paper.

        inputs is 3D, factored=True/False: Each row in the last dimension is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of sha
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector [i, j].

        Arguments:
            inputs (np.ndarray): An array of input vectors in rows. See above for behavior.
            factored (bool): See above for behavior.
        """
        if len(inputs.shape) == 2:
            if factored:
                return self.sess.run(
                    [self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac],
                    feed_dict={self.sy_pred_in2d: inputs}
                )
            else:
                return self.sess.run(
                    [self.sy_pred_mean2d, self.sy_pred_var2d],
                    feed_dict={self.sy_pred_in2d: inputs}
                )
        else:
            return self.sess.run(
                [self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac],
                feed_dict={self.sy_pred_in3d: inputs}
            )

    def create_prediction_tensors(self, inputs, factored=False, *args, **kwargs):
        """See predict() above for documentation.
        """
        
        rew_mean, rew_logvar, state_mean, state_logvar = self.get_weighted_mean_logvar(
            self._compile_output_layer(inputs))
        factored_mean = tf.concat([rew_mean, state_mean], axis = -1)
        factored_variance = tf.math.exp(tf.concat([rew_logvar, state_logvar], axis = -1))

        if inputs.shape.ndims == 2 and not factored:
            mean = tf.reduce_mean(factored_mean, axis=0)
            variance = tf.reduce_mean(tf.square(factored_mean - mean), axis=0) + \
                       tf.reduce_mean(factored_variance, axis=0)
            return mean, variance
        return factored_mean, factored_variance

    def save_model(self, _dir, epoch):
        
        model_dir = os.path.join(_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        val_dict = {}
        for var in (self.nonoptvars + self.optvars):
            val_dict[var.name] = self.sess.run(var)
        pickle.dump(val_dict, open(os.path.join(model_dir, 'epoch_'+str(epoch)+'.pkl'), 'wb'))

    def load_model(self, _file):

        model_var_vals = pickle.load(open(_file, 'rb'))
        for var in (self.nonoptvars + self.optvars):
            self.sess.run(tf.assign(var, model_var_vals[var.name]))

    def _load_structure(self):
        """Uses the saved structure in self.model_dir with the name of this network to initialize
        the structure of this network.
        """
        structure = []
        with open(os.path.join(self.model_dir, "%s.nns" % self.name), "r") as f:
            for line in f:
                kwargs = {
                    key: val for (key, val) in
                    [argval.split("=") for argval in line[3:-2].split(", ")]
                }
                kwargs["input_dim"] = int(kwargs["input_dim"])
                kwargs["output_dim"] = int(kwargs["output_dim"])
                kwargs["weight_decay"] = None if kwargs["weight_decay"] == "None" else float(kwargs["weight_decay"])
                kwargs["activation"] = None if kwargs["activation"] == "None" else kwargs["activation"][1:-1]
                kwargs["ensemble_size"] = int(kwargs["ensemble_size"])
                structure.append(FC(**kwargs))
        self.layers = structure

    #######################
    # Compilation methods #
    #######################
    def get_weighted_mean_logvar(self, cur_out):
        dim_output = self.layers[-1].get_output_dim()
        mean, logvar = cur_out[:, :, :dim_output // 2], cur_out[:, :, dim_output // 2:]

        rew_mean = mean[:,:,:1] * self.reward_prediction_weight
        state_mean = mean[:,:,1:]

        rew_logvar = self._incorporate_max_min_logvar(logvar[:,:,:1] + np.log(self.reward_prediction_weight ** 2),
                                                      self.max_rew_logvar, self.min_rew_logvar)
        
        state_logvar = self._incorporate_max_min_logvar(logvar[:,:,1:], self.max_state_logvar, self.min_state_logvar)

        return rew_mean, rew_logvar, state_mean, state_logvar

    def _incorporate_max_min_logvar(self, logvar, max_logvar, min_logvar):
        logvar = max_logvar - tf.nn.softplus(max_logvar - logvar)
        return min_logvar + tf.nn.softplus(logvar - min_logvar)

    def _compile_output_layer(self, inputs, ret_log_var=False):
        """Compiles the output of the network at the given inputs.

        If inputs is 2D, returns a 3D tensor where output[i] is the output of the ith network in the ensemble.
        If inputs is 3D, returns a 3D tensor where output[i] is the output of the ith network on the ith input matrix.

        Arguments:
            inputs: (tf.Tensor) A tensor representing the inputs to the network
            ret_log_var: (bool) If True, returns the log variance instead of the variance.

        Returns: (tf.Tensors) The mean and variance/log variance predictions at inputs for each network
            in the ensemble.
        """
        cur_out = self.scaler.transform(inputs)
        for layer in self.layers:
            cur_out = layer.compute_output_tensor(cur_out)

        return cur_out
        # return self.get_weighted_mean_logvar(curr_out)

        # if ret_log_var:
        #     return mean, logvar
        # else:
        #     return mean, tf.exp(logvar)

    def _get_model_loss_dict(self, inputs,  targets, tiled_contexts, add_state_pred=True, add_rew_pred=True):
        total_model_loss = 0
        loss_dict = self._compile_losses(inputs, targets, tiled_contexts)
        if add_rew_pred:
            total_model_loss += loss_dict['total_rew_loss']
        if add_state_pred:
            total_model_loss += loss_dict['total_state_loss']
        loss_dict['total_model_loss'] = total_model_loss
        return loss_dict

    def _compile_losses(self, inputs, targets, tiled_contexts):

        inputs = tf.concat([inputs, tiled_contexts], axis=-1)
        rew_mean, rew_logvar, state_mean, state_logvar = self.get_weighted_mean_logvar(
            self._compile_output_layer(inputs))
        total_rew_loss, mse_rew_loss = self._compile_log_likelihood(rew_mean, rew_logvar, targets[:,:, :1],
                                                                    self.min_rew_logvar, self.max_rew_logvar)
        total_state_loss, mse_state_loss = self._compile_log_likelihood(state_mean, state_logvar, targets[:,:, 1:],
                                                                        self.min_state_logvar, self.max_state_logvar)

        return OrderedDict({'total_rew_loss': total_rew_loss,
                            'total_state_loss': total_state_loss,
                            'mse_rew_loss': mse_rew_loss,
                            'mse_state_loss': mse_state_loss})

    def _compile_log_likelihood(self, mean, log_var, targets, min_logvar, max_logvar):
        """Helper method for compiling the loss function.

        The loss function is obtained from the log likelihood, assuming that the output
        distribution is Gaussian, with both mean and (diagonal) covariance matrix being determined
        by network outputs.

        Arguments:
            inputs: (tf.Tensor) A tensor representing the input batch
            targets: (tf.Tensor) The desired targets for each input vector in inputs.
            inc_var_loss: (bool) If True, includes log variance loss.

        Returns: (tf.Tensor) A tensor representing the loss on the input arguments.
        """

        inv_var = tf.exp(-log_var)
        mse_losses = tf.reduce_mean(tf.square(mean - targets), axis=-1)
        # if inc_var_loss:
        mean_losses = tf.reduce_mean(tf.square(mean - targets) * inv_var, axis=-1)
        var_losses = tf.reduce_mean(log_var, axis=-1)

        total_losses = mean_losses + var_losses + tf.add_n(self.decays) + \
                       0.01 * tf.reduce_sum(max_logvar) - 0.01 * tf.reduce_sum(min_logvar)

        return tf.reduce_mean(total_losses), tf.reduce_mean(mse_losses)
