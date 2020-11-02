from .customgymenv import CustomGymEnv

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))

from gym import spaces
from utils import PARTICLE_SIZES
import numpy as np

class LightDarkEnv(CustomGymEnv):
    def __init__(self):
        super(LightDarkEnv, self).__init__(
                os.path.dirname(os.path.realpath(__file__)) +'/../../cpp/build/GymEnvLightDark',
                100, PARTICLE_SIZES['LightDark'])

        self.action_space = spaces.Box(np.array([-1, -1, 0]), np.array([1, 1, 1]), dtype=np.float32)
