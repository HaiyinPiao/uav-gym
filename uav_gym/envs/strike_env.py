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
        Type: Discrete(2)
        Num	Action
        0	Straight Flight
        1	Turn Left
        2   Turn Right
    """
    def __init__(self):
        self.uav = uav_t()
        self.obstacle = obstacle_t()

        high = np.array([math.pi,
                         math.pi/2.0,
                         np.finfo(np.float32).max,
                         ARENA_X_LEN*10.0,
                         ARENA_Y_LEN*10.0],
                        dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.uav.avail_phi))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.state = None
    
    def step(self, action):
        _x = self.uav.x
        self.state = self.uav.step(action)
        x_ = self.uav.x
        reward = -0.5
        if(x_>_x):
            reward+=0.5
        done = False
        if self.uav.x>4000 and abs(self.uav.y)<1000:
            reward += 1000
            done = True
        # done = True if self._x>10 else False
        if self.obstacle.is_in_range([self.uav.x,self.uav.y]):
            reward -= 400
            done = True
        # done = False
        # done = True if ((self.uav.x>4000 and abs(self.uav.y)<1000) or abs(self.uav.x>ARENA_X_LEN) or abs(self.uav.y>ARENA_X_LEN) or self.obstacle.is_in_range([self.uav.x,self.uav.y])) else False
        # done = True if (self.uav.x>4000 and abs(self.uav.y)<1000) or abs(self.uav.x)>ARENA_X_LEN or abs(self.uav.y)>ARENA_X_LEN else False
        # if done:
        #     reward -=300
        

        return np.array(self.state), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(5,))
        # state variables
        self.phi = 0.0
        self.psi = 90.0*DEG2RAD
        self.psi_dot = 0.0
        self.x = 0#-ARENA_X_LEN/2.0+1000
        self.y = 0.0

        self.state = self.uav.reset()

        return np.array(self.state)
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass