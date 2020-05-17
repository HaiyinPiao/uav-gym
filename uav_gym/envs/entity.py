from .strike_args import *
import numpy as np

# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

class entity_t():
    def __init__(self):
        self.alive = True
        pass
    
    def step(self, action):
        pass
    def is_movable(self):
        pass
    def is_alive(self):
        pass
    def get_pos(self):
        return np.array([self.x,self.y])
    def get_r(self):
        pass
    def kill(self):
        self.alive = False

class uav_t(entity_t):
    def __init__(self):
        self.v = 100.0
        self.phi_dot_lim = 28.0*DEG2RAD
        self.phi_lim = 77.0*DEG2RAD
        self.alive = True
        
        # state variables
        self.phi = 0.0
        self.psi = 0.0*DEG2RAD
        self.phi_dot = 0.0
        self.psi_dot = 0.0
        self.x = 0#-ARENA_X_LEN/2.0+1000
        self.y = 0.0

        # action mapping
        self.avail_phi = np.arange(-70.0,70.0,10.0)
        self.avail_phi *= DEG2RAD
        # print(self.avail_phi)

        self.reset()

        # self.x_traj = []
        # self.y_traj = []
    
    def step(self, action):
        """
        action=0: -70 phi
        action=1: -60 phi
        action=2: -50 phi
        ... ...
        action=13: +70 phi
        """
        # if action>0:
        #     sign = -1.0 if action is 1 else 1.0
        #     # print(self.phi)
        #     self.phi += sign*self.phi_dot_lim*TAU
        #     self.phi = np.clip(self.phi, -self.phi_lim, self.phi_lim)
        #     # print(self.phi)
        phi_c = self.avail_phi[action]
        sign = -1.0 if phi_c<0.0 else 1.0
        self.phi += sign*self.phi_dot_lim*TAU
        if sign>0:
            self.phi = np.clip(self.phi, 0, phi_c)
        else:
            self.phi = np.clip(self.phi, phi_c, 0)

        self.psi_dot = (GRAVITY/self.v)*math.tan(self.phi)
        self.psi += self.psi_dot*TAU
        self.x += self.v*math.sin(self.psi)*TAU
        self.y += self.v*math.cos(self.psi)*TAU
        # print(self.x, self.y)
        # self.x_traj.append(self.x)
        # self.y_traj.append(self.y)
        reward = 0.0
        done = False
        state = (float(self.alive), self.phi, self.psi_dot, self.psi, self.x, self.y) if self.is_alive() else (0.0,0.0,0.0,0.0,0.0,0.0)

        return state, reward, done, {}

    def reset(self, phi=0.0, psi_dot=0.0, psi=0.0*DEG2RAD, x=0.0, y=0.0):
        self.alive = True
        self.phi = phi
        self.psi_dot = psi_dot
        self.psi = psi
        self.x = x
        self.y = y

        self.state = (float(self.alive), self.phi, self.psi_dot, self.psi, self.x, self.y)

        return self.state

    def is_movable(self):
        return True

    def is_alive(self):
        return self.alive
    def get_r(self):
        return 0.0

class obstacle_t(entity_t):
    def __init__(self):
        self.alive = True
        self.x = 0.0
        self.y = 0.0
        self.r = 0.0
    
    def step(self, uavs:[]):
        reward = [0.0] *len(uavs)
        done = False
        for v,i in zip(uavs,range(len(uavs))):
            if self.is_alive and self.is_in_range([v.x,v.y]):
                reward[i] += 500
                # self.alive = False
                done = True
        state = (float(self.alive), self.x, self.y, self.r) if self.is_alive() else (0.0, 0.0, 0.0, 0.0)

        return state, reward, done, {}

    def reset(self, x:float, y:float, r:float):
        self.alive = True
        self.x = x
        self.y = y
        self.r = r

        self.state = (float(self.alive), self.x, self.y, self.r)

        return self.state

    def is_movable(self):
        return False

    def is_in_range(self, pos:[]):
        assert(len(pos)==2)
        pivot = [self.x,self.y]
        range = math.sqrt(sum([(a - b)**2 for (a,b) in zip(pos,pivot)]))
        return True if range<self.r else False

    def is_alive(self):
        return self.alive
    
    def get_r(self):
        return self.r


if __name__ == "__main__":
    uav = uav_t()
    for i in range(300):
        if i<70:
            action = 0
        elif i>=70 and i<140:
            action = 3
        else:
            action = 13
        uav.step(action)

    entities = []
    entities.append(uav_t())
    entities.append(obstacle_t())

    # for e in entities:
    #     print(e.is_movable())

    # print(entities[1].is_in_range([3999,0]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.gca(projection='3d')

    # set figure information
    # ax.set_title("3D_Curve")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")

    # draw the figure, the color is r = read
    plt.axis([-5000,5000,-5000,5000])
    plt.plot(uav.x_traj,uav.y_traj)

    circle = mpathes.Circle(np.array([2500,2500]),500,facecolor= 'red', alpha=0.3)
    ax.add_patch(circle)

    plt.show()

    pass