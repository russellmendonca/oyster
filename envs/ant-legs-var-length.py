import numpy as np
from .ant import AntEnv
import itertools
from . import register_env
@register_env('ant-legs-var-length')
class AntLegsVarLength(AntEnv):

    def __init__(self, n_tasks):

        self.tasks = self.get_tasks(n_tasks)

        super(AntLegsVarLength, self).__init__()

        self.unit_geom_size = np.round_(np.sqrt(2),7)/10
        self.init_body_pos  = self.model.body_pos.copy()
        self.init_geom_pos  = self.model.geom_pos.copy()

    def get_tasks(self, n_tasks):
        assert n_tasks == 81
        all_tasks = []
        #train tasks : thigh_multiplier fixed to be 1.4, ankle multiplier can be 0.7, 1.4 or 2.1
        factors = [0.7, 1.4, 2.1]
        for ankle_config in itertools.product(factors, factors, factors, factors):
            all_tasks.append(dict(
                thigh_multipliers=np.ones(4),
                ankle_multipliers= ankle_config
            ))
        return all_tasks

    def set_leg(self, leg_idx, leg_config):
        """leg : index of the leg being set
            leg_config : dictionary with lengths of thigh and ankle

            Note : Ankle geom pos is divided by 2, because in the default xml
            the length of the ankle is twice the unit length"""

        thigh_geom_idx = 3 + 3 * leg_idx
        ankle_geom_idx = thigh_geom_idx + 1
        knee_body_idx = 4 + 3 * leg_idx

        self.model.geom_size[thigh_geom_idx, 1] = self.unit_geom_size *leg_config[0]
        self.model.geom_size[ankle_geom_idx, 1] = self.unit_geom_size * leg_config[1]
        self.model.body_pos[knee_body_idx, :] = self.init_body_pos[knee_body_idx] * leg_config[0]

        self.model.geom_pos[thigh_geom_idx, :] = self.init_geom_pos[thigh_geom_idx] * leg_config[0]
        self.model.geom_pos[ankle_geom_idx, :] = (self.init_geom_pos[ankle_geom_idx] / 2) * leg_config[1]

    def reset_task(self, idx):

        self.sim.reset()
        task = self.tasks[idx]

        for leg_index, leg_config in enumerate(zip(task['thigh_multipliers'], task['ankle_multipliers'])):
            self.set_leg(leg_index, leg_config)


if __name__ == '__main__':

    env =  AntLegsVarLength(81)

    for idx in range(81):
        env.reset()
        env.reset_task(idx)

        for _ in range(100):
            env.step(np.zeros(env.action_space.shape))
            #import ipdb; ipdb.set_trace()
            env.render()
