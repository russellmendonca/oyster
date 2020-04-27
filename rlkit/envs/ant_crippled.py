import os
import numpy as np
from .ant import AntEnv
from . import register_env

@register_env('ant-crippled')
class AntCrippledEnv(AntEnv):

    def __init__(self, n_tasks=4):
        assert n_tasks == 4
        super(AntCrippledEnv, self).__init__()

        self.tasks = np.arange(4)
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

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def reset_task(self, idx):

        self.crippled_leg = idx
        self.sim.reset()
        self.cripple_mask = np.ones(self.action_space.shape)
        # Pick which actuators to disable
        if self.crippled_leg == 0:
            self.cripple_mask[2] = 0
            self.cripple_mask[3] = 0
        elif self.crippled_leg == 1:
            self.cripple_mask[4] = 0
            self.cripple_mask[5] = 0
        elif self.crippled_leg == 2:
            self.cripple_mask[6] = 0
            self.cripple_mask[7] = 0
        elif self.crippled_leg == 3:
            self.cripple_mask[0] = 0
            self.cripple_mask[1] = 0

        # Make the removed leg look red
        geom_rgba = self._init_geom_rgba.copy()
        if self.crippled_leg == 0:
            geom_rgba[3, :3] = np.array([0, 1, 0])
            geom_rgba[4, :3] = np.array([0, 1, 0])
        elif self.crippled_leg == 1:
            geom_rgba[6, :3] = np.array([1, 0, 1])
            geom_rgba[7, :3] = np.array([1, 0, 1])
        elif self.crippled_leg == 2:
            geom_rgba[9, :3] = np.array([1, 1, 0])
            geom_rgba[10, :3] = np.array([1, 1, 0])
        elif self.crippled_leg == 3:
            geom_rgba[12, :3] = np.array([1, 0, 0])
            geom_rgba[13, :3] = np.array([1, 0, 0])

        self.model.geom_rgba[:] = geom_rgba

        # Make the removed leg not affect anything
        temp_size = self._init_geom_size.copy()
        temp_pos = self._init_geom_pos.copy()

        if self.crippled_leg == 0:
            # Top half
            temp_size[3, 0] = temp_size[3, 0] / 2
            temp_size[3, 1] = temp_size[3, 1] / 2
            # Bottom half
            temp_size[4, 0] = temp_size[4, 0] / 2
            temp_size[4, 1] = temp_size[4, 1] / 2
            temp_pos[4, :] = temp_pos[3, :]

        elif self.crippled_leg == 1:
            # Top half
            temp_size[6, 0] = temp_size[6, 0] / 2
            temp_size[6, 1] = temp_size[6, 1] / 2
            # Bottom half
            temp_size[7, 0] = temp_size[7, 0] / 2
            temp_size[7, 1] = temp_size[7, 1] / 2
            temp_pos[7, :] = temp_pos[6, :]

        elif self.crippled_leg == 2:
            # Top half
            temp_size[9, 0] = temp_size[9, 0] / 2
            temp_size[9, 1] = temp_size[9, 1] / 2
            # Bottom half
            temp_size[10, 0] = temp_size[10, 0] / 2
            temp_size[10, 1] = temp_size[10, 1] / 2
            temp_pos[10, :] = temp_pos[9, :]

        elif self.crippled_leg == 3:
            # Top half
            temp_size[12, 0] = temp_size[12, 0] / 2
            temp_size[12, 1] = temp_size[12, 1] / 2
            # Bottom half
            temp_size[13, 0] = temp_size[13, 0] / 2
            temp_size[13, 1] = temp_size[13, 1] / 2
            temp_pos[13, :] = temp_pos[12, :]

        self.model.geom_size[:] = temp_size
        self.model.geom_pos[:] = temp_pos

if __name__ == '__main__':
    env = AntCrippledEnv()
    for idx in range(4):
        env.reset()
        env.reset_task(idx)
        for _ in range(100):
            env.step(env.action_space.sample())
            env.render()
