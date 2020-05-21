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
        self.n_agents = N_AGENTS
        self.uavs = [uav_t() for _ in range(self.n_agents)]
        self.targets = [obstacle_t() for _ in range(N_TARGETS)]

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
        # calc only for agent[0]
        wingmans = copy.deepcopy(self.uavs)
        wingmans.pop(-1)
        for _ in wingmans:
            high = np.concatenate((high, rel_high))
        for _ in self.targets:
            high = np.concatenate((high, rel_high))
                        
        self.action_space = [spaces.Discrete(len(self.uavs[0].avail_phi)) for _ in range(self.n_agents)]
        self.observation_space = [spaces.Box(-high, high, dtype=np.float32) for _ in range(self.n_agents)]
        self.seed()
        self.state = [None] * self.n_agents
        self.steps = 0
        self.render = RENDER

    def _get_rel_states(self, v:uav_t, state:np.array):
        # for v in self.uavs:
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
        reward = [0.0] * self.n_agents
        for _ in range(A_REPEAT):
            state = [np.array([])] * self.n_agents
            done = [False] * self.n_agents
            # calc uavs native observations
            for i in range(self.n_agents):
                s, _, _, _ = self.uavs[i].step(action[i])
                # uav native observation
                state[i] = np.concatenate((state[i], np.array(s)))
                if self.uavs[i].is_alive():
                    reward[i] -= 0.2
                
            # calc target status
            for t in self.targets:
                s, r, d, _ = t.step(self.uavs)
                if t.is_alive():
                    reward = [i + j for i, j in zip(reward, r)]
                    if d:
                        t.kill()

            # relative observations
            for i in range(self.n_agents):
                state[i] = self._get_rel_states(self.uavs[i], state[i])
            for i in range(self.n_agents): 
                self.state[i] = state[i].tolist()

            # judge if all targets eliminated
            all_tgts_clr = [False] * len(self.targets)
            for i in range(len(self.targets)):
                all_tgts_clr[i] = False if self.targets[i].is_alive() else True
            level_clr = [True] * self.n_agents if all(all_tgts_clr) else [False] * self.n_agents
            
            # judge if *ANY* drone fly Out-Of-Border(OOB)
            uav_out = [False] * self.n_agents
            for i in range(self.n_agents):
                if self.uavs[i].is_alive() and (abs(self.uavs[i].x)>ARENA_X_LEN or abs(self.uavs[i].y)>ARENA_Y_LEN):
                    self.uavs[i].kill()
                    reward[i] -= 200.0
                    uav_out[i] = True

            # judge if uavs die(die condition indluding OOB).
            uav_die = [False] * self.n_agents
            for i in range(self.n_agents):
                uav_die[i] = False if self.uavs[i].is_alive() else True
            uav_die = [d or o for d, o in zip(uav_die, uav_out)]

            self.steps += 1
            # judge if expsode ended
            episode_len_exceed = [True]*self.n_agents if self.steps>=MAX_EPS_LEN else [False]*self.n_agents
                    
            # all together
            # done = level_clr or uav_die or episode_len_exceed
            done = [c or d or e for c, d, e in zip(level_clr, uav_die, episode_len_exceed)]

            # uavs trajectories logging
            self.vis.log_uavs(self.uavs)
            # if self.steps%200==0:
            #     self.vis.log_tgts(self.targets)

            if all(done):
                break
        # print(self.state[1][4],self.state[1][5])
        return self.state, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        state = [np.array([])] * self.n_agents
        for i in range(self.n_agents):
            if POS_FIXED:
                s = np.array(self.uavs[i].reset())
            else:
                s = np.array(self.uavs[i].reset(psi=np.random.randint(-3,3),x=np.random.randint(-ARENA_X_LEN/5.0,ARENA_X_LEN/5.0),y=np.random.randint(-ARENA_Y_LEN/5.0,ARENA_Y_LEN/5.0)))
            state[i] = np.concatenate((state[i], np.array(s)))

        if POS_FIXED:
            s = self.targets[0].reset(3000,3000,1000)
            s = self.targets[1].reset(-3000,-3000,1000)
            s = self.targets[2].reset(0,2000,1000)
        else:
            for t in self.targets:
                s = t.reset(np.random.randint(-ARENA_X_LEN/2.5,ARENA_X_LEN/2.5),np.random.randint(-ARENA_Y_LEN/2.5,ARENA_Y_LEN/2.5),800)

        # relative observations
        for i in range(self.n_agents):
            state[i] = self._get_rel_states(self.uavs[i], state[i])
        for i in range(self.n_agents): 
            self.state[i] = state[i].tolist()

        self.vis = plot_t(self.uavs, self.targets)
        self.steps = 0
        
        self.vis.log_tgts(self.targets)

        return self.state
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__ == "__main__":
    env = StrikeEnv()
    pass