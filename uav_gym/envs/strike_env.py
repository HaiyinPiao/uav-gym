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
        self.n_agents = 2
        self.uavs = [uav_t() for _ in range(self.n_agents)]
        self.targets = [obstacle_t() for _ in range(4)]

        uav_high = np.array([1.0,math.pi,
                         math.pi/2.0,
                         np.finfo(np.float32).max,
                         ARENA_X_LEN*10.0,
                         ARENA_Y_LEN*10.0],
                        dtype=np.float32)
        # alive, IFF, AO, r, r_lethal
        rel_high = np.array([1.0,1.0,
                         math.pi,
                         ARENA_Y_LEN*2.0,
                         ARENA_Y_LEN*2.0],
                        dtype=np.float32)

        high = np.array([])
        # uav native observations
        # for v in self.uavs:
        high = np.concatenate((high, uav_high))
        # relative observation calculations
        for v in self.uavs:
            wingmans = copy.deepcopy(self.uavs)
            wingmans.remove(wingmans[self.uavs.index(v)])
            for _ in wingmans:
                high = np.concatenate((high, rel_high))
            for _ in self.targets:
                high = np.concatenate((high, rel_high))
                        
        self.action_space = [spaces.Discrete(len(self.uavs[0].avail_phi)) for _ in range(self.n_agents)]
        self.observation_space = [spaces.Box(-high, high, dtype=np.float32) for _ in range(self.n_agents)]
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

    def step(self, action:[]):
        # self.state = self.uav.step(action)
        state = [np.array([])] * self.n_agents
        reward = [0.0] * self.n_agents
        done = [False] * self.n_agents
        
        for _ in range(A_REPEAT):
            for i in range(self.n_agents):
                reward[i] -= 0.1
                tgts_status = []

                for t in self.targets:
                    s, r, clr, _ = t.step(self.uavs[i])
                    if t.is_alive():
                        reward[i] += r
                        if clr:
                            t.kill()
                    tgts_status.append(t.is_alive())

                # for v,a in zip(self.uavs,action):
                # native observation
                s, _, _, _ = v.step(action[i])
                state[i] = np.concatenate((state[i], np.array(s)))
                # relative observations
                state[i] = self._get_rel_states(state[i])
                
                # TODO
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

        return self.state, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        state = np.array([])
        for v in self.uavs:
            if POS_FIXED:
                s = np.array(v.reset())
            else:
                s = np.array(v.reset(psi=np.random.randint(-3,3),x=np.random.randint(-ARENA_X_LEN/5.0,ARENA_X_LEN/5.0),y=np.random.randint(-ARENA_Y_LEN/5.0,ARENA_Y_LEN/5.0)))
            state = np.concatenate((state, np.array(s)))

        if POS_FIXED:
            s = self.targets[0].reset(3000,3000,1000)
            # state = np.concatenate((state, np.array(s)))
            s = self.targets[1].reset(-3000,-3000,1000)
            # state = np.concatenate((state, np.array(s)))
            s = self.targets[2].reset(0,2000,1000)
            # state = np.concatenate((state, np.array(s)))
        else:
            for t in self.targets:
                s = t.reset(np.random.randint(-ARENA_X_LEN/2.5,ARENA_X_LEN/2.5),np.random.randint(-ARENA_Y_LEN/2.5,ARENA_Y_LEN/2.5),500)
                # state = np.concatenate((state, np.array(s)))

        # relative observations
        state = self._get_rel_states(state)

        self.state = state.tolist()
        self.vis = plot_t(self.uavs, self.targets)
        self.steps = 0

        return self.state
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__ == "__main__":
    env = StrikeEnv()
    pass