#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='SAC Benchmarking Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--num-env', type=int, default=10,
                    help='Number of environments (default: 10)')
parser.add_argument('--model-path', required=True,
                    help='Path to model file.')
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))

from models import RLActorNet
from utils import Statistics
from utils import PARTICLE_SIZES
import gymenvs

import gym
import multiprocessing
import numpy as np
import queue
import struct
import time
import torch
import zmq

TASK = args.task
POLICY_MODEL_PATH = args.model_path
PARTICLE_SIZE = PARTICLE_SIZES[TASK]

NUM_ENVIRONMENTS = args.num_env
ZMQ_ADDRESS = 'tcp://127.0.0.1'

def environment_process(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    env = gym.make(TASK + '-v0')
    policy = RLActorNet(PARTICLE_SIZE, TASK in ['LightDark', 'IntentionTag']).cuda().float()
    policy.load_state_dict(torch.load(POLICY_MODEL_PATH))
    policy.eval()

    while True:
        total_reward = 0
        best_tracking_error = None
        steps = 0
        state = env.reset()
        while True:
            with torch.no_grad():
                action = policy.sample(torch.tensor(state, dtype=torch.float).cuda().unsqueeze(0))[0].cpu().numpy()[0]
            state, reward, done, tracking_error = env.step(action)
            total_reward += reward
            best_tracking_error = tracking_error if best_tracking_error is None else min(best_tracking_error, tracking_error)
            steps += 1

            if done:
                socket.send_pyobj((steps, total_reward, 0 if reward > 0 else 1, best_tracking_error))
                socket.recv_pyobj()
                break


if __name__ == '__main__':

    # Prepare zmq server.
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port(ZMQ_ADDRESS)

    # Start processes.
    processes = [multiprocessing.Process(target=environment_process, args=(port,), daemon=True) for i in range(NUM_ENVIRONMENTS)]
    for p in processes:
        p.start()

    steps_statistics = Statistics()
    total_reward_statistics = Statistics()
    collision_statistics = Statistics()
    tracking_error_statistics = Statistics()

    while True:
        (steps, total_reward, collision, best_tracking_error) = socket.recv_pyobj()
        socket.send_pyobj(None)

        steps_statistics.append(steps)
        collision_statistics.append(collision)
        total_reward_statistics.append(total_reward)
        tracking_error_statistics.append(best_tracking_error)

        print('\r\033[K{}\t| {}\t| {}\t| {}'.format(
            steps_statistics,
            total_reward_statistics,
            collision_statistics,
            tracking_error_statistics), end="")
