import os
os.environ['OMP_NUM_THREADS'] = '1' # Multithreading doesn't really help for our pipeline.
import cv2
cv2.setNumThreads(1) # Multithreading doesn't really help for our pipeline.

import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from models import MAGICGenNet, RLActorNet
from utils import Statistics, PARTICLE_SIZES

from base64 import b64encode, b64decode
from collections import namedtuple
from controller import Controller
from data_store import DataClient
from perception import BOARD_HEIGHT, BOARD_WIDTH
from subprocess import Popen, PIPE
import multiprocessing
import numpy as np
import struct
import sys
import time
import torch
import signal

NUM_PARTICLES = 100
MOVE_DISTANCE = 100
HANDCRAFTED = False
GOAL = (1058, 428)
GOAL_RADIUS = 80
NOISY_REGIONS = [
    [(590, 40), (700, 558)],
    [(790, 40), (900, 558)],
]
BOT_START = (300, 299)
BALL_START = (450, 299)
IS_SAC = True

last_draw_time = None

class Environment:

    def __init__(self, macro_length, handcrafted):
        l = ['../../cpp/build/RealWorldPuckPushDespot']
        l.append('--macro-length={}'.format(macro_length))
        if handcrafted:
            l.append('--handcrafted')
        self.process = Popen(l, shell=False, stdout=PIPE, stdin=PIPE)
        self.macro_length = macro_length

    def initialize_belief(self, state):
        self.process.stdin.write('INITIALIZE_BELIEF\n'.encode('utf8'))
        self.process.stdin.write('{}\n'.format(state[0]).encode('utf8'))
        self.process.stdin.write('{}\n'.format(state[1]).encode('utf8'))
        if state[2] is None:
            self.process.stdin.write('\n'.encode('utf8'))
        else:
            self.process.stdin.write('{}\n'.format(state[2]).encode('utf8'))
        if state[3] is None:
            self.process.stdin.write('\n'.encode('utf8'))
        else:
            self.process.stdin.write('{}\n'.format(state[3]).encode('utf8'))
        self.process.stdin.flush()
        self.process.stdout.readline()

    def sgpc_get_belief(self):
        self.process.stdin.write('GET_BELIEF\n'.encode('utf8'))
        self.process.stdin.flush()
        state_raw = np.array([x[0] for x in struct.iter_unpack('f', b64decode(self.process.stdout.readline().decode('utf8').strip()))])
        return state_raw.astype(np.float32).reshape((NUM_PARTICLES, -1))

    def sgpc_invoke_planner(self, params):
        self.process.stdin.write('INVOKE_PLANNER\n'.encode('utf8'))
        params_raw = b64encode(b''.join(struct.pack('f', p) for p in params))
        self.process.stdin.write((params_raw.decode('ascii') + '\n').encode('utf8'))
        self.process.stdin.flush()
        return np.array([float(self.process.stdout.readline().decode('utf8').strip()) for _ in range(self.macro_length + 3)])

    # Action: (stationary?, orientation)
    # Observation: (bot_position.x, bot_position.y, ball_position.x, ball_position.y)
    def update_belief(self, action, observation):
        print('Belief udpate: action = {}, observation = {}'.format(action, observation))
        self.process.stdin.write('UPDATE_BELIEF\n'.encode('utf8'))
        self.process.stdin.write('{}\n'.format(action).encode('utf8'))
        self.process.stdin.write('{}\n'.format(observation[0]).encode('utf8'))
        self.process.stdin.write('{}\n'.format(observation[1]).encode('utf8'))
        if observation[2] is None:
            self.process.stdin.write('\n'.encode('utf8'))
        else:
            self.process.stdin.write('{}\n'.format(observation[2]).encode('utf8'))
        if observation[3] is None:
            self.process.stdin.write('\n'.encode('utf8'))
        else:
            self.process.stdin.write('{}\n'.format(observation[3]).encode('utf8'))
        self.process.stdin.flush()
        self.process.stdout.readline()

    def sample_belief(self, count):
        self.process.stdin.write('SAMPLE_BELIEF\n'.encode('utf8'))
        self.process.stdin.write('{}\n'.format(count).encode('utf8'))
        self.process.stdin.flush()
        return np.stack([np.array([float(self.process.stdout.readline().decode('utf8').strip()) for _ in range(4)]) for _ in range(count)])

    def shutdown(self):
        self.process.stdin.write('SHUTDOWN\n'.encode('utf8'))
        self.process.stdin.flush()
        self.process.kill()


def main():
    # Read script parameters.
    if not IS_SAC:
        macro_length = int(sys.argv[1])
        if len(sys.argv) > 2:
            model_path = sys.argv[2]
        else:
            model_path = None
    else:
        macro_length = 1
        model_path = sys.argv[1]

    print('Loading model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not IS_SAC:
        gen_model = MAGICGenNet(PARTICLE_SIZES['PuckPush']).to(device).float()
        if model_path is not None:
            gen_model.load_state_dict(torch.load(model_path))
    else:
        actor_model = RLActorNet(PARTICLE_SIZES['PuckPush'], False).to(device).float()
        if model_path is not None:
            actor_model.load_state_dict(torch.load(model_path))

    controller = Controller()
    dc = DataClient()


    cv2.namedWindow("frame")
    cv2.moveWindow("frame", int((1980 - BOARD_WIDTH) / 2), int((1080 - BOARD_HEIGHT) / 2))

    print('Waiting for perception...')
    while dc['perception'] is None:
        time.sleep(0.01)
    p = dc['perception']
    while p['bot'] is None:
        time.sleep(0.01)
        p = dc['perception']

    print('Initializing belief...')
    env = Environment(macro_length, HANDCRAFTED)
    env.initialize_belief([*p['bot'][0], *(p['ball'][0] if p['ball'] is not None else (None, None))])

    belief_samples = []
    macro_action = None
    macro_action_start = None

    def update_belief(action):
        p = dc['perception']

        bot = p['bot']
        if bot is None:
            bot = (None, None)
        else:
            bot = bot[0]

        ball = p['ball']
        if ball is None:
            ball = (None, None)
        else:
            ball = ball[0]

        env.update_belief(action, [*bot, *ball])


    def draw():
        global last_draw_time
        if last_draw_time is None or time.time() - last_draw_time > 1.0 / 30:
            p = dc['perception']
            annotated_frame = dc['frame']

            for region in NOISY_REGIONS:
                annotated_frame = cv2.rectangle(annotated_frame, region[0], region[1], (62, 134, 149), 1)
            annotated_frame = cv2.circle(annotated_frame, GOAL, GOAL_RADIUS, (0, 255, 0), 1)

            for sampled_state in belief_samples:
                annotated_frame = cv2.drawMarker(annotated_frame, tuple(np.int0(sampled_state[0:2])), (255, 255, 0), cv2.MARKER_CROSS, 5, 1)
                annotated_frame = cv2.drawMarker(annotated_frame, tuple(np.int0(sampled_state[2:4])), (0, 255, 255), cv2.MARKER_CROSS, 5, 1)

            if macro_action is not None:
                s = np.copy(macro_action_start)
                for a in macro_action:
                    annotated_frame = cv2.line(annotated_frame,
                            tuple(np.int0(s)), tuple(np.int0(s + MOVE_DISTANCE * np.array([np.cos(a), np.sin(a)]))),
                            (0, 255, 128), 2)
                    s += MOVE_DISTANCE * np.array([np.cos(a), np.sin(a)])

            annotated_frame = cv2.drawMarker(annotated_frame, BOT_START, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 2)
            annotated_frame = cv2.drawMarker(annotated_frame, BALL_START, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 2)

            if p['bot'] is not None:
                annotated_frame = cv2.drawMarker(annotated_frame, tuple(np.int0(p['bot'][0])), (255, 0, 255), cv2.MARKER_CROSS, 15, 2)

            if p['ball'] is not None:
                annotated_frame = cv2.drawMarker(annotated_frame, tuple(np.int0(p['ball'][0])), (255, 0, 255), cv2.MARKER_CROSS, 15, 2)


            cv2.imshow('frame', annotated_frame)
            cv2.waitKey(1)
            last_draw_time = time.time()



    print('Displaying initial belief. Starting in...')

    belief_samples = env.sample_belief(500)
    draw()
    cv2.waitKey(100)

    for i in range(5):
        print(5 - i)
        time.sleep(1)

    total_steps = 0
    while True:
        sgpc_belief = env.sgpc_get_belief()
        with torch.no_grad():
            if not IS_SAC:
                params = gen_model(torch.tensor(sgpc_belief, dtype=torch.float32, device=device).unsqueeze(0))[0].squeeze(0).cpu().numpy()
                response = env.sgpc_invoke_planner(params)
                macro_action = response[:macro_length]
                macro_action_start = dc['perception']['bot'][0]
                (num_nodes, depth, value) = response[macro_length:]
                print('Action = {}'.format(macro_action))
                print('Num nodes = {}'.format(num_nodes))
                print('Depth {}'.format(depth))
                print('Value = {}'.format(value))

                for action in macro_action:
                    print('Steps = {}'.format(total_steps))
                    total_steps += 1
                    controller.turn(action, draw)
                    controller.move(MOVE_DISTANCE, draw)
                    update_belief(action)
                    belief_samples = env.sample_belief(500)
            else:
                action = actor_model.sample(torch.tensor(sgpc_belief, dtype=torch.float32, device=device).unsqueeze(0))[2].squeeze(0).cpu().numpy()
                action = np.arctan2(action[1], action[0])
                macro_action = [action] # For drawing
                macro_action_start = dc['perception']['bot'][0]
                print('Action = {}'.format(action))
                print('Steps = {}'.format(total_steps))
                total_steps += 1
                controller.turn(action, draw)
                controller.move(MOVE_DISTANCE, draw)
                update_belief(action)
                belief_samples = env.sample_belief(500)


    env.shutdown()

if __name__ == '__main__':
    main()
