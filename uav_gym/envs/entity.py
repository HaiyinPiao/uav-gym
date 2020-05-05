from .strike_args import *
import numpy as np

# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

class entity_t():
    def __init__(self):
        pass
    
    def step(self, action):
        pass
    def is_movable(self):
        return False

class uav_t(entity_t):
    def __init__(self):
        self.v = 100.0
        self.phi_dot_lim = 28.0*DEG2RAD
        self.phi_lim = 77.0*DEG2RAD
        
        # state variables
        self.phi = 0.0
        self.psi = 90.0*DEG2RAD
        self.phi_dot = 0.0
        self.psi_dot = 0.0
        self.x = 0#-ARENA_X_LEN/2.0+1000
        self.y = 0.0

        self.reset()

        self.x_traj = []
        self.y_traj = []
    
    def step(self, action):
        """
        action=0: straight flight
        action=1: turn left
        action=2: turn right
        """
        if action>0:
            sign = -1.0 if action is 1 else 1.0
            # print(self.phi)
            self.phi += sign*self.phi_dot_lim*TAU
            self.phi = np.clip(self.phi, -self.phi_lim, self.phi_lim)
            print(self.phi)
        self.psi_dot = (GRAVITY/self.v)*math.tan(self.phi)
        self.psi += self.psi_dot*TAU
        self.x += self.v*math.sin(self.psi)*TAU
        self.y += self.v*math.cos(self.psi)*TAU
        # print(self.x, self.y)
        self.x_traj.append(self.x)
        self.y_traj.append(self.y)

        return (self.phi, self.psi_dot, self.psi, self.x, self.y)

    def reset(self):
        self.phi = 0.0
        self.psi = 90.0*DEG2RAD
        self.psi_dot = 0.0
        self.x = 0#-ARENA_X_LEN/2.0+1000
        self.y = 0.0

        self.state = (self.phi, self.psi_dot, self.psi, self.x, self.y)

        return self.state

    def is_movable(self):
        return True

class obstacle_t(entity_t):
    def __init__(self):
        self.x = 2500.0
        self.y = 0.0
        self.r = 1500.0
    
    def step(self, action):
        pass

    def is_movable(self):
        return False

    def is_in_range(self, pos:[]):
        assert(len(pos)==2)
        pivot = [self.x,self.y]
        range = math.sqrt(sum([(a - b)**2 for (a,b) in zip(pos,pivot)]))
        return True if range<self.r else False


if __name__ == "__main__":
    uav = uav_t()
    for i in range(300):
        if i<70:
            action = 0
        elif i>=70 and i<140:
            action = 1
        else:
            action = 2
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

    circle = mpathes.Circle(np.array([2500,0]),1500,facecolor= 'red', alpha=0.3)
    ax.add_patch(circle)


    plt.show()

    pass