from .customgymenv import CustomGymEnv

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))

from gym import spaces
from utils import PARTICLE_SIZES
import numpy as np

class PuckPushEnv(CustomGymEnv):
    def __init__(self):
        super(PuckPushEnv, self).__init__(
                os.path.dirname(os.path.realpath(__file__)) +'/../../cpp/build/GymEnvPuckPush',
                100, PARTICLE_SIZES['PuckPush'])

        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)

