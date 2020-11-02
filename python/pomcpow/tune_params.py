#!/usr/bin/env python3

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))

from environment import Environment, Response
from utils import Statistics

import argparse
import itertools
import multiprocessing
import numpy as np
import queue
import struct
import time
import torch
import zmq

parser = argparse.ArgumentParser(description='POMCPOW Parameters Tuning Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--trials', type=int, default=50,
                    help='Number of trials (default: 50)')
parser.add_argument('--num-env', type=int, default=50,
                    help='Number of environments (default: 50)')
args = parser.parse_args()

TASK = args.task

NUM_ENVIRONMENTS = args.num_env
ZMQ_ADDRESS = 'tcp://127.0.0.1'
TEST_RUNS = args.trials

# These ranges are selected to approximately reflect the range of parameters
# used in the POMCPOW paper.
UCB = [25, 50, 75, 100]
K_ACTION = [12.5, 25.0, 37.5, 50.0]
ALPHA_ACTION = [0.0250, 0.050, 0.075, 0.100]
K_OBSERVATION = [2.5, 5.0, 7.5, 10.0]
ALPHA_OBSERVATION = [0.025, 0.050, 0.075, 0.100]
PARAMS = [UCB, K_ACTION, ALPHA_ACTION, K_OBSERVATION, ALPHA_OBSERVATION]

def environment_process(ucb,
        k_action, alpha_action,
        k_observation, alpha_observation,
        port, remaining, lock):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    environment = Environment(TASK, False,
            ucb, k_action, alpha_action, k_observation, alpha_observation)

    while True:
        lock.acquire()
        r = remaining.value
        if r >= 1:
            remaining.value -= 1
        lock.release()
        if r == 0:
            break

        while True:
            steps = 0
            total_reward = 0
            collision = None
            num_nodes = 0
            depth = 0
            tracking_error = None
            while True:
                # Read step response.
                response = environment.process_response()

                # Add to statistics.
                steps += response.steps
                total_reward += response.reward
                num_nodes = max(num_nodes, response.num_nodes)
                depth = max(depth, response.depth)
                tracking_error = response.tracking_error if tracking_error is None else min(tracking_error, response.tracking_error)
                if response.is_terminal:
                    if response.reward < 0:
                        collision = 1
                    else:
                        collision = 0
                    break
            socket.send_multipart([
                struct.pack('i', steps),
                struct.pack('f', total_reward),
                struct.pack('i', collision),
                struct.pack('i', num_nodes),
                struct.pack('i', depth),
                struct.pack('f', tracking_error)])
            socket.recv_multipart()

def test(ucb, k_action, alpha_action, k_observation, alpha_observation):
    # Prepare zmq server.
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port(ZMQ_ADDRESS)

    # Start processes.
    remaining = multiprocessing.Value('i', TEST_RUNS, lock=False)
    lock = multiprocessing.RLock()
    processes = [multiprocessing.Process(
                target=environment_process,
                args=(ucb, k_action, alpha_action, k_observation, alpha_observation,
                    port, remaining, lock),
                daemon=True)
            for i in range(NUM_ENVIRONMENTS)]
    for p in processes:
        p.start()

    total_reward_statistics = Statistics()

    completed = 0
    while completed < TEST_RUNS:
        # Read request and process.
        data = socket.recv_multipart()
        (steps, total_reward, collision, num_nodes, depth, tracking_error) = \
                (struct.unpack(f, d)[0] for (f, d) in zip(['i', 'f', 'i', 'i', 'i', 'f'], data))
        total_reward_statistics.append(total_reward)
        socket.send_multipart([b''])
        completed += 1

    # Destroy processes.
    for p in processes:
        p.terminate()
        p.join()
    socket.close()

    return total_reward_statistics.mean()

if __name__ == '__main__':
    best_params = None
    best_result = None

    i = 0
    for params in itertools.product(*PARAMS):
        result = test(*params)
        print('Tested: {} / {}'.format(i + 1, np.prod([len(p) for p in PARAMS])))
        print('Current: {}, {}, {}, {}, {} -> {}'.format(*params, result))
        if best_result is None or result > best_result:
            best_params = params
            best_result = result
        print('Best: {}, {}, {}, {}, {} -> {}'.format(*best_params, best_result))
        i += 1

