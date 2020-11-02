from base64 import b64encode, b64decode
from collections import namedtuple
from subprocess import Popen, PIPE
import numpy as np
import struct
import time

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

    def __init__(self, task, macro_length, visualize):
        l = ['../../cpp/build/DespotHandcraftedEnv{}'.format(task)]
        l.append('--macro-length={}'.format(macro_length))
        if visualize:
            l.append('--visualize')
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
