import numpy as np
import tensorflow as tf

from models.fc import FC
from models.bnn import BNN

def construct_model(obs_dim, act_dim, context_dim = 5 ,  rew_dim=1, hidden_dim=200,
					ada_state_dynamics_pred = True, ada_rew_pred = True,
					fast_adapt_steps = 2 , fast_adapt_lr = 0.01,
					reg_weight = 1, pred_dynamics = True, fixed_preupdate_context = True,  num_networks=1, num_elites=1, session=None):
	print('[ BNN ] Observation dim {} | Action dim: {} | Hidden dim: {}'.format(obs_dim, act_dim, hidden_dim))
	params = {'name': 'BNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session}
	output_dim = rew_dim + obs_dim
	model = BNN(params, context_dim , obs_dim, act_dim, ada_state_dynamics_pred, ada_rew_pred,
				fast_adapt_steps , fast_adapt_lr, reg_weight , fixed_preupdate_context )
	model.add(FC(hidden_dim, input_dim=obs_dim+act_dim+context_dim, activation="swish", weight_decay=0.000025))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))

	model.add(FC(output_dim, weight_decay=0.0001))
	model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
	return model

def format_samples_for_training(samples):
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	rew = samples['rewards']
	delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((rew, delta_obs), axis=-1)
	return inputs, outputs

def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))

if __name__ == '__main__':
	model = construct_model()
