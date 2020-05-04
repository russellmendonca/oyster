import numpy as np
from .ant import AntEnv
from . import register_env
@register_env('ant-wind')
class AntWindEnv(AntEnv):

    def __init__(self, n_tasks, wind_mag=3):
        self.wind_mag=wind_mag
        self.tasks = self.sample_tasks(n_tasks)
        super(AntWindEnv, self).__init__()
        self.model.opt.density = 1.2 #density of air

    def sample_tasks(self, n_tasks):
        assert n_tasks == 120
        np.random.seed(42)
        train_tasks =  [(self.wind_mag*np.cos(theta), self.wind_mag*np.sin(theta))
                     for theta in np.random.uniform(0, 1.5*np.pi, (100))]
        test_tasks = [(self.wind_mag * np.cos(theta), self.wind_mag * np.sin(theta))
                       for theta in np.linspace(1.5*np.pi, 2 * np.pi, (20))]
        return np.concatenate([train_tasks, test_tasks])

    def reset_task(self, idx):
        self.model.opt.wind[:2] = self.tasks[idx]
        self.reset()

if __name__ == '__main__':

    env =  AntWindEnv(120)
    for idx in range(120):
        env.reset()
        env.reset_task(idx)
        for _ in range(100):
            env.step(np.zeros(env.action_space.shape))
            env.render()
