import signal
from base64 import b64encode, b64decode
from subprocess import Popen, PIPE
import numpy as np
import struct
import time
from collections import namedtuple
import os

Response = namedtuple('Response', [
    'best_value',
    'steps',
    'reward',
    'is_terminal',
    'is_failure',
    'num_nodes',
    'depth',
    'stats'
])

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

        self.process = None
        self.process = Popen(l, shell=False, stdout=PIPE, stdin=PIPE, preexec_fn=os.setsid)

    def __del__(self):
        if self.process is not None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

    def process_response(self):
        best_value = float(self.process.stdout.readline().decode('utf8').strip())
        if best_value > 9999999:
            best_value = None
        steps = int(self.process.stdout.readline().decode('utf8').strip())
        reward = float(self.process.stdout.readline().decode('utf8').strip())
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

        return Response(best_value, steps, reward,
                is_terminal, is_failure,
                num_nodes, depth, stats)
