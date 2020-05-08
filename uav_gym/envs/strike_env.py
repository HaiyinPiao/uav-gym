import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .strike_args import *
from .entity import *

class StrikeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    """
    Observation: 
        Type: Box(4)
        Num	Observation     Min             Max
        0	phi             -4.8            4.8
        1	psi_dot         -Inf            Inf
        2	psi             -24 deg         24 deg
        3	x               -Inf            Inf
        4	y               -Inf            Inf
        
    Actions:
        ref class uav_t(entity_t).step(self, action)
        relative observation
    """
    def __init__(self):
        self.uavs = [uav_t() for _ in range(1)]
        self.targets = [obstacle_t() for _ in range(3)]


        uav_high = np.array([1.0,math.pi,
                         math.pi/2.0,
                         np.finfo(np.float32).max,
                         ARENA_X_LEN*10.0,
                         ARENA_Y_LEN*10.0],
                        dtype=np.float32)
        tgt_high = np.array([1.0,
                         ARENA_X_LEN*2.0,
                         ARENA_Y_LEN*2.0,
                         ARENA_Y_LEN],
                        dtype=np.float32)

        high = np.array([])
        for v in self.uavs:
            high = np.concatenate((high, uav_high))
        for t in self.targets:
            high = np.concatenate((high, tgt_high))
                        
        self.action_space = spaces.Discrete(len(self.uavs[0].avail_phi))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.state = None
    
    def step(self, action):
        # self.state = self.uav.step(action)
        state = np.array([])
        for v in self.uavs:
            s = v.step(action)
            state = np.concatenate((self.state, np.array(s)))
        
        reward = -0.5
        done = False
        # if self.uav.x>4000 and abs(self.uav.y)<1000:
        #     reward += 1000
        #     done = True

        for v in self.uavs:
            for t in self.targets:
                if t.is_alive:
                    if t.is_in_range([v.x,v.y]):
                        reward+=200
                        self.targets.alive = False
                else:
                    

        self.state = state.to_list()

        return state, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(5,))
        # # state variables
        # self.phi = 0.0
        # self.psi = 90.0*DEG2RAD
        # self.psi_dot = 0.0
        # self.x = 0#-ARENA_X_LEN/2.0+1000
        # self.y = 0.0

        # self.state = np.concatenate((self.state, np.array(s)))
        state = np.array([])
        for v in self.uavs:
            s = np.array(v.reset())
            state = np.concatenate((state, np.array(s)))

        s = self.targets[0].reset(3000,3000,500)
        state = np.concatenate((state, np.array(s)))
        s = self.targets[0].reset(-3000,-3000,500)
        state = np.concatenate((state, np.array(s)))
        s = self.targets[0].reset(0,2000,500)
        state = np.concatenate((state, np.array(s)))

        self.state = state.to_list()

        return state
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__ == "__main__":
    env = StrikeEnv()
    pass