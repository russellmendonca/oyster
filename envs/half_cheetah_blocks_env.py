import os
import numpy as np
from .mujoco_env import MujocoEnv
from . import register_env

@register_env('cheetah-blocks')
class HalfCheetahBlocksEnv(MujocoEnv):

    def __init__(self, n_tasks):

        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                              "assets", "half_cheetah_blocks.xml")
        super(HalfCheetahBlocksEnv, self).__init__(
            model_path=model_path, model_path_is_local=False,
            automatically_set_obs_and_action_space=True)

        self.tasks = self.sample_tasks(n_tasks)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten()[9:],
            self.data.qvel.flat[8:],
            self.get_body_com("torso").flat,
        ])

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost)
        return (observation, reward, done, infos)

    def reset_task(self, idx):
        damping = self.model.dof_damping.copy()
        damping[:8] = self.tasks[idx]
        self.model.dof_damping[:] = damping

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        return np.random.uniform(0,5, size = (num_tasks, 8))

if __name__ == '__main__':
    n_tasks = 10
    env = HalfCheetahBlocksEnv(n_tasks)
    for idx in range(n_tasks):
        env.reset()
        env.reset_task(idx)
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()
