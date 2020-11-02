from base64 import b64encode, b64decode
from gym import spaces
from subprocess import Popen, PIPE
import gym
import numpy as np
import struct

class CustomGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': []}

    def __init__(self, env_path, num_particles, particle_size):
        super(CustomGymEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(num_particles, particle_size), dtype=np.float32)

        self.process = Popen([env_path], shell=False, stdout=PIPE, stdin=PIPE)
        self.num_particles = num_particles
        self.particle_size = particle_size

    def step(self, action):
        self.process.stdin.write('STEP\n'.encode('utf8'))
        for f in action:
            self.process.stdin.write((str(f) + '\n').encode('utf8'))
        self.process.stdin.flush()

        observation = self._read_state()
        reward = float(self.process.stdout.readline().decode('utf8').strip())
        is_terminal = int(self.process.stdout.readline().decode('utf8').strip()) == 1
        tracking_error = float(self.process.stdout.readline().decode('utf8').strip())

        return observation, reward, is_terminal, tracking_error

    def reset(self):
        self.process.stdin.write('RESET\n'.encode('utf8'))
        self.process.stdin.flush()
        return self._read_state()

    def render(self, mode):
        raise NotImplementedError()

    def _read_state(self):
        state_raw = np.array([x[0] for x in struct.iter_unpack('f', b64decode(self.process.stdout.readline().decode('utf8').strip()))])
        return state_raw.astype(np.float32).reshape((self.num_particles, self.particle_size))
