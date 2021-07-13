from base64 import b64encode, b64decode
from collections import namedtuple
from subprocess import Popen, PIPE
import numpy as np
import struct
import time

Response = namedtuple('Response', [
    'best_value',
    'steps',
    'discounted_reward',
    'undiscounted_reward',
    'is_terminal',
    'is_failure',
    'num_nodes',
    'depth',
    'stats'
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
        if best_value > 9999999:
            best_value = None
        steps = int(self.process.stdout.readline().decode('utf8').strip())
        discounted_reward = float(self.process.stdout.readline().decode('utf8').strip())
        undiscounted_reward = float(self.process.stdout.readline().decode('utf8').strip())
        is_terminal = float(self.process.stdout.readline().decode('utf8').strip()) > 0.5
        is_failure = float(self.process.stdout.readline().decode('utf8').strip()) > 0.5
        num_nodes = int(self.process.stdout.readline().decode('utf8').strip())
        if num_nodes > 9999999:
            num_nodes = None
        depth = int(self.process.stdout.readline().decode('utf8').strip())
        if depth > 9999999:
            depth = None

        stats_str = [self.process.stdout.readline().decode('utf8').strip() for _ in range(5)]
        stats = []
        for s in stats_str:
            if s == '':
                stats.append(None)
            else:
                stats.append(np.array([x[0] for x in struct.iter_unpack('f', b64decode(s))]).astype(np.float32))

        return Response(best_value, steps, discounted_reward, undiscounted_reward,
                is_terminal, is_failure, num_nodes, depth, stats)
