from base64 import b64encode, b64decode
from subprocess import Popen, PIPE
import numpy as np
import os
import signal
import struct
import time
import torch

NUM_PARTICLES = 100

class Environment:

    def __init__(self, macro_length):
        l = ['../../cpp/build/SummitEnvDriveHard']
        l.append('--macro-length={}'.format(macro_length))
        self.process = None
        self.process = Popen(l, shell=False, stdout=PIPE, stdin=PIPE, preexec_fn=os.setsid)
        #self.params_log_file = open('sync/params.log', 'w')

    def __del__(self):
        if self.process is not None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

        '''
        if self.params_log_file is not None:
            self.params_log_file.close()
        '''

    def initialize_belief(self, obs, context):
        self.process.stdin.write('INITIALIZE_BELIEF\n'.encode('utf8'))
        self.process.stdin.write((b64encode(b''.join(struct.pack('f', s) for s in obs)).decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.write((b64encode(b''.join(struct.pack('f', s) for s in context)).decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.flush()
        self.process.stdout.readline()

    def update_belief(self, action, obs):
        self.process.stdin.write('UPDATE_BELIEF\n'.encode('utf8'))
        self.process.stdin.write((b64encode(b''.join(struct.pack('f', s) for s in action)).decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.write((b64encode(b''.join(struct.pack('f', s) for s in obs)).decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.flush()
        self.process.stdout.readline()

    def pop_update_belief(self):
        self.process.stdin.write('POP_UPDATE_BELIEF\n'.encode('utf8'))
        self.process.stdin.flush()
        return self.process.stdout.readline().decode('utf8').strip() # Returns 'TERMINAL' if resulting belief is terminal.

    def sample_belief(self):
        self.process.stdin.write('SAMPLE_BELIEF\n'.encode('utf8'))
        self.process.stdin.flush()

        # Read state.
        state_raw = self.process.stdout.readline().decode('utf8').strip()
        state = np.array([x[0] for x in struct.iter_unpack('f', b64decode(state_raw))]).astype(np.float32)
        return state


    def debug_belief(self, obs):
        self.process.stdin.write('DEBUG_BELIEF\n'.encode('utf8'))
        self.process.stdin.write((b64encode(b''.join(struct.pack('f', s) for s in obs)).decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.flush()
        self.process.stdout.readline()

    def forward_plan_handcrafted(self, macro_action):
        self.process.stdin.write('FORWARD_PLAN_HANDCRAFTED\n'.encode('utf8'))
        for action in macro_action:
            self.process.stdin.write((b64encode(b''.join(struct.pack('f', s) for s in action)).decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.flush()
        self.process.stdout.readline()

    def forward_plan_magic(self, macro_action, model):
        self.process.stdin.write('FORWARD_PLAN_MAGIC\n'.encode('utf8'))
        for action in macro_action:
            self.process.stdin.write((b64encode(b''.join(struct.pack('f', s) for s in action)).decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.flush()

        forward_result = self.process.stdout.readline().decode('utf8').strip()
        if isinstance(forward_result, str) and forward_result == 'TERMINAL':
            return 'TERMINAL'

        # Read context.
        context_raw = self.process.stdout.readline().decode('utf8').strip()
        context = np.array([x[0] for x in struct.iter_unpack('f', b64decode(context_raw))]).astype(np.uint8)

        # Read state.
        state_raw = self.process.stdout.readline().decode('utf8').strip()
        state = np.array([x[0] for x in struct.iter_unpack('f', b64decode(state_raw))]).astype(np.float32)

        # Invoke model.
        params = model(context, state)

        # Write params.
        params_raw = b64encode(b''.join(struct.pack('f', p) for p in params))
        #self.params_log_file.write(params_raw.decode('ascii') + '\n')
        self.process.stdin.write((params_raw.decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.flush()

        # Read return.
        self.process.stdout.readline()

    def forward_plan_pomcpow(self, macro_action):
        self.process.stdin.write('FORWARD_PLAN_POMCPOW\n'.encode('utf8'))
        for action in macro_action:
            self.process.stdin.write((b64encode(b''.join(struct.pack('f', s) for s in action)).decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.flush()
        self.process.stdout.readline()

    def pop_forward_plan_result(self):
        self.process.stdin.write('POP_FORWARD_PLAN_RESULT\n'.encode('utf8'))
        self.process.stdin.flush()

        result_raw = self.process.stdout.readline().decode('utf8').strip()
        if result_raw == 'TERMINAL': # Returns 'TERMINAL' if forwarded belief is terminal.
            return 'TERMINAL'
        else:
            macro_action_raw = np.array([x[0] for x in struct.iter_unpack('f', b64decode(result_raw))])
            depth = int(self.process.stdout.readline().decode('utf8').strip())
            return (macro_action_raw.astype(np.float32).reshape((-1, 2)), depth)

    def terminate(self):
        self.process.stdin.write('TERMINATE\n'.encode('utf8'))
        self.process.stdin.flush()
