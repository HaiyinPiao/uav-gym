import math
import numpy as np
import gym

from .strike_args import *
# from .strike_env import *
from .entity import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

class plot_t():
    def __init__(self, uavs:[], tgts:[]):
        self.uavs_traj = [[] for _ in range(len(uavs))]
        self.tgts_circs = [{'x': t.x, 'y': t.y, 'r': t.r} for t in tgts]
        # print(self.tgts_circs)

    def log(self, uavs:[], tgts:[]):
        assert(isinstance(uavs[0], uav_t))
        assert(isinstance(tgts[0], obstacle_t))
        for v, trj in zip(uavs,self.uavs_traj):
            trj.append([v.x,v.y])

    def plot(self):
        fig = plt.figure()
        plt.axis('equal')
        ax = fig.add_subplot(111)
        plt.axis([-ARENA_X_LEN/2,ARENA_X_LEN/2,-ARENA_Y_LEN/2,ARENA_Y_LEN/2])
        # print(self.tgts_circs)
        for t in self.tgts_circs:
            circle = mpathes.Circle(np.array([t["x"],t["y"]]),t["r"],facecolor= 'red', alpha=0.3)
            ax.add_patch(circle)

        for trj in self.uavs_traj:
            trj = np.array(trj)
            # print(trj.shape)
            plt.plot(trj[:,0].tolist(),trj[:,1].tolist())
        plt.show()
        pass