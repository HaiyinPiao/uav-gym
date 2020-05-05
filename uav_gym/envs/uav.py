from strike_args import *
import numpy as np

class entity_t():
    def __init__(self):
        pass
    
    def step(self, action):
        pass

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
            # print(self.phi)
        self.psi_dot = (GRAVITY/self.v)*math.tan(self.phi)
        self.psi += self.psi_dot*TAU
        self.x += self.v*math.sin(self.psi)*TAU
        self.y += self.v*math.cos(self.psi)*TAU
        print(self.x, self.y)

if __name__ == "__main__":
    uav = uav_t()
    for _ in range(500):
        uav.step(2)
    pass