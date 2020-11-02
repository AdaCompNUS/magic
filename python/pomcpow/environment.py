from base64 import b64encode, b64decode
from subprocess import Popen, PIPE
import numpy as np
import struct
import time
from collections import namedtuple

Response = namedtuple('Response', [
    'best_value',
    'steps',
    'reward',
    'is_terminal',
    'num_nodes',
    'depth',
    'tracking_error'
])

NUM_PARTICLES = 100

class Environment:

    def __init__(self, task, visualize,
            ucb=None, k_action=None, alpha_action=None, k_observation=None, alpha_observation=None):
        l = ['../../cpp/build/PomcpowEnv{}'.format(task)]
        if visualize:
            l.append('--visualize')
        if ucb is not None:
            l.append('--ucb={}'.format(ucb))
        if k_action is not None:
            l.append('--k_action={}'.format(k_action))
        if alpha_action is not None:
            l.append('--alpha_action={}'.format(alpha_action))
        if k_observation is not None:
            l.append('--k_observation={}'.format(k_observation))
        if alpha_observation is not None:
            l.append('--alpha_observation={}'.format(alpha_observation))
        self.process = Popen(l, shell=False, stdout=PIPE, stdin=PIPE)

    def process_response(self):
        best_value = float(self.process.stdout.readline().decode('utf8').strip())
        steps = int(self.process.stdout.readline().decode('utf8').strip())
        reward = float(self.process.stdout.readline().decode('utf8').strip())
        is_terminal = int(self.process.stdout.readline().decode('utf8').strip()) == 1
        num_nodes = int(self.process.stdout.readline().decode('utf8').strip())
        depth = int(self.process.stdout.readline().decode('utf8').strip())
        tracking_error = float(self.process.stdout.readline().decode('utf8').strip())

        return Response(best_value, steps, reward, is_terminal, num_nodes, depth, tracking_error)
