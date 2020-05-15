import math
import numpy as np
import gym
import copy
from gym import error, spaces, utils
from gym.utils import seeding

from .strike_args import *
from .entity import *
from .relative_obs import *
from .logplot import *

class StrikeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    """ 
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
        # alive, x, y, r_lethal
        tgt_high = np.array([1.0,
                         ARENA_X_LEN*2.0,
                         ARENA_Y_LEN*2.0,
                         ARENA_Y_LEN],
                        dtype=np.float32)
        # IFF, AO, r, r_lethal
        rel_high = np.array([1.0,
                         math.pi,
                         ARENA_Y_LEN*2.0,
                         ARENA_Y_LEN*2.0],
                        dtype=np.float32)

        high = np.array([])
        # uav and targets native observations
        for v in self.uavs:
            high = np.concatenate((high, uav_high))
        for t in self.targets:
            high = np.concatenate((high, tgt_high))
        # relative observation calculations
        for v in self.uavs:
            wingmans = copy.deepcopy(self.uavs)
            wingmans.remove(wingmans[self.uavs.index(v)])
            for _ in wingmans:
                high = np.concatenate((high, rel_high))
            for _ in self.targets:
                high = np.concatenate((high, rel_high))
                        
        self.action_space = spaces.Discrete(len(self.uavs[0].avail_phi))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.state = None

        self.steps = 0

    def _get_rel_states(self, state:np.array):
        for v in self.uavs:
            wingmans = copy.deepcopy(self.uavs)
            wingmans.remove(wingmans[self.uavs.index(v)])
            for u in wingmans:
                s = calc_rel_obs(v,u)
                state = np.concatenate((state, np.array(s)))
            for t in self.targets:
                s = calc_rel_obs(v,t)
                state = np.concatenate((state, np.array(s)))  
        return state     

    def step(self, action):
        # self.state = self.uav.step(action)
        reward = 0.0
        done = False
        
        for _ in range(A_REPEAT):
            state = np.array([])
            for v in self.uavs:
                s, _, _, _ = v.step(action)
                state = np.concatenate((state, np.array(s)))
            
            reward -= 0.1
            # if self.uav.x>4000 and abs(self.uav.y)<1000:
            #     reward += 1000
            #     done = True
            tgts_status = []

            for t in self.targets:
                s, r, clr, _ = t.step(self.uavs)
                if t.is_alive():
                    reward += r
                    if clr:
                        t.kill()
                    state = np.concatenate((state, np.array(s)))
                else:
                    # zeros nest state while alive==false
                    s0 = np.zeros_like(s)
                    state = np.concatenate((state, np.array(s0)))
                tgts_status.append(t.is_alive())

            # relative observations
            state = self._get_rel_states(state)
                        
            self.state = state.tolist()

            # done judgement
            # judge if all targets eliminated
            all_tgts_clr = tgts_status.count(False) is len(self.targets)
            
            # judge if *ANY* drone fly Out-Of-Border(OOB)
            OOBs = []
            for v in self.uavs:
                OOBs.append(abs(v.x)>ARENA_X_LEN or abs(v.y)>ARENA_Y_LEN)
            any_uav_out = OOBs.count(True)>0
            reward -= any_uav_out*200.0

            self.steps += 1
            episode_len_exceed = True if self.steps>=MAX_EPS_LEN else False
                    
            done = all_tgts_clr or any_uav_out or episode_len_exceed

            # uavs trajectories logging
            self.vis.log(self.uavs, self.targets)

            if done:
                break

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
            if POS_FIXED:
                s = np.array(v.reset())
            else:
                s = np.array(v.reset(x=np.random.randint(-ARENA_X_LEN/5.0,ARENA_X_LEN/5.0),y=np.random.randint(-ARENA_Y_LEN/5.0,ARENA_Y_LEN/5.0)))
            state = np.concatenate((state, np.array(s)))         

        if POS_FIXED:
            s = self.targets[0].reset(3000,3000,1000)
            state = np.concatenate((state, np.array(s)))
            s = self.targets[1].reset(-3000,-3000,1000)
            state = np.concatenate((state, np.array(s)))
            s = self.targets[2].reset(0,2000,1000)
            state = np.concatenate((state, np.array(s)))
        else:
            for t in self.targets:
                s = t.reset(np.random.randint(-ARENA_X_LEN/2.0,ARENA_X_LEN/2.0),np.random.randint(-ARENA_Y_LEN/2.0,ARENA_Y_LEN/2.0),1000)
                state = np.concatenate((state, np.array(s)))

        # relative observations
        state = self._get_rel_states(state)

        self.state = state.tolist()
        self.vis = plot_t(self.uavs, self.targets)
        self.steps = 0

        return state
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__ == "__main__":
    env = StrikeEnv()
    pass