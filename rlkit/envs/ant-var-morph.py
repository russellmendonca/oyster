import os
import numpy as np
from .mujoco_env import MujocoEnv
from . import register_env

@register_env('ant-var-morph')
class AntVarMorph(MujocoEnv):

    def __init__(self, n_tasks=32):
        assert n_tasks == 32

        super(AntVarMorph, self).__init__(
            model_path='ant.xml', model_path_is_local=True,
            automatically_set_obs_and_action_space=True
        )
        self.tasks = np.arange(n_tasks)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.crippled_leg = 0

    '''
    our "front" is in +x direction, to the right side of screen
    LEG 4 (they call this back R)
    action0: front-right leg, top joint 
    action1: front-right leg, bottom joint

    LEG 1 (they call this front L)
    action2: front-left leg, top joint
    action3: front-left leg, bottom joint 

    LEG 2 (they call this front R)
    action4: back-left leg, top joint
    action5: back-left leg, bottom joint 

    LEG 3 (they call this back L)
    action6: back-right leg, top joint
    action7: back-right leg, bottom joint 
    geom_names has 
            ['floor','torso_geom',
            'aux_1_geom','left_leg_geom','left_ankle_geom', --1
            'aux_2_geom','right_leg_geom','right_ankle_geom', --2
            'aux_3_geom','back_leg_geom','third_ankle_geom', --3
            'aux_4_geom','rightback_leg_geom','fourth_ankle_geom'] --4
    '''

    def step(self, a):
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = torso_velocity[0] / self.dt
        ctrl_cost = 0.  # .5 * np.square(a).sum()
        contact_cost = 0
        survive_reward = 0.05  # 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def reset_task(self, idx):
        #TODO: implement!
        pass


if __name__ == '__main__':
    env = AntVarMorph()
    for idx in range(4):
        env.reset()
        env.reset_task(idx)
        for _ in range(100):
            env.step(env.action_space.sample())
            env.render()
