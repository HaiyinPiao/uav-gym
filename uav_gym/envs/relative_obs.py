import math
from .math import *
import numpy as np
import cmath
from .entity import *


def calc_vec_angle(v0, v1):
    ang = math.atan2(v1[1], v1[0]) - math.atan2(v0[1], v0[0])
    ang = clip_psi(ang)
    return ang

def calc_AO(v:uav_t, t:entity_t):
    v0 = cmath.rect(1.0, v.psi)
    v0 = np.array([v0.real,v0.imag])
    los = t.get_pos()-v.get_pos()
    v1 = los/np.linalg.norm(los)
    return calc_vec_angle(v0, v1)

def calc_r(v:entity_t, t:entity_t):
    return np.linalg.norm(t.get_pos()-v.get_pos())

def calc_rel_obs(v:entity_t, t:entity_t):
    # IFF, AO, r, r_lethal
    state = None
    if v.is_alive() and t.is_alive():
        foe = True if isinstance (t,obstacle_t) else False 
        AO = calc_AO(v,t)
        r = calc_r(v,t)
        r_lethal = t.get_r()
        state = (float(t.is_alive()),float(foe), AO, r, r_lethal)
    else:
        state = (0.0,0.0,0.0,0.0,0.0)
    return state

if __name__ == "__main__":
    rhs = calc_vec_angle(np.array([1,0]),np.array([-1,-0.1]))
    print(rhs)
    pass