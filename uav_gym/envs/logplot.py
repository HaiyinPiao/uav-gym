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
        # self.d_tgt = {'x', 'y', 'r'}
        self.tgts_circs = [[] for _ in range(len(tgts))]
        # print(self.tgts_circs)

    def log_uavs(self, uavs:[]):
        assert(isinstance(uavs[0], uav_t))
        for v, trj in zip(uavs,self.uavs_traj):
            trj.append([v.x,v.y])

    def log_tgts(self, tgts:[]):
        assert(isinstance(tgts[0], obstacle_t))
        for t, trj in zip(tgts,self.tgts_circs):
            trj.append({'x':t.x, 'y':t.y, 'r':t.r})

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axis('equal')
        plt.axis([-ARENA_X_LEN/1.5,ARENA_X_LEN/1.5,-ARENA_Y_LEN/1.5,ARENA_Y_LEN/1.5])
        # print(self.tgts_circs)
        border = plt.Rectangle((-ARENA_X_LEN/2,-ARENA_Y_LEN/2),ARENA_X_LEN,ARENA_Y_LEN,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(border)
        for tgt_i in self.tgts_circs:
            alpha_delta = 0.02
            for t in tgt_i:
                # circle = mpathes.Circle(np.array([t["x"],t["y"]]),t["r"],facecolor='red',alpha=0.01+alpha_delta)
                circle = mpathes.Circle(np.array([t["x"],t["y"]]),t["r"],facecolor='red',alpha=0.2)
                alpha_delta += alpha_delta
                ax.add_patch(circle)

        for trj in self.uavs_traj:
            trj = np.array(trj)
            # print(trj.shape)
            plt.plot(trj[:,0].tolist(),trj[:,1].tolist())
        plt.show()
        pass