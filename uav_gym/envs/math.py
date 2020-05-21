import numpy as np
import math

def clip_psi(psi):
    if psi > math.pi:
        psi = -(math.pi-(psi-math.pi))
    elif psi < -math.pi:
        psi = math.pi+(psi+math.pi)
    return psi