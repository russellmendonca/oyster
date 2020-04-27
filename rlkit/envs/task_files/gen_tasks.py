import pickle
import numpy as np


def cheetah_vel():
    tasks = [{"velocity": velocity} for velocity in np.linspace(0, 3, 10)]
    pickle.dump(tasks, open('cheetah-vel/unif-0-3-10tasks.pkl', 'wb'))

# cheetah_vel()
