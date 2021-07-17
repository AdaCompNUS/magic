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
parser.add_argument('--trials', type=int, default=30,
                    help='Number of trials (default: 30)')
parser.add_argument('--num-env', type=int, default=16,
                    help='Number of environments (default: 16)')
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

def environment_process(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    while True:
        socket.send_pyobj(('REQ_PARAMS',))
        params = socket.recv_pyobj()
        if isinstance(params, str) and params == 'TERMINAL':
            break
        environment = Environment(TASK, False, *params)

        total_reward = 0
        while True:
            # Read step response.
            response = environment.process_response()

            # Add to statistics.
            total_reward += response.reward
            if response.is_terminal:
                break

        socket.send_pyobj([
            'ADD_RESULT',
            params,
            total_reward])
        socket.recv_pyobj()

if __name__ == '__main__':

    # Prepare zmq server.
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port(ZMQ_ADDRESS)

    # Start processes.
    processes = [multiprocessing.Process(target=environment_process, args=(port,), daemon=True) for i in range(NUM_ENVIRONMENTS)]
    for p in processes:
        p.start()


    params = list(itertools.product(*PARAMS))
    param_results = dict()
    for param in params:
        param_results[param] = []

    queue = []
    for param in params:
        for _ in range(TEST_RUNS):
            queue.append(param)

    total_count = len(queue)
    completed_count = 0
    best_result = None
    while True:
        request = socket.recv_pyobj()
        instruction = request[0]
        data = request[1:]

        if instruction == 'REQ_PARAMS':
            if len(queue) == 0:
                socket.send_pyobj('TERMINAL')
            else:
                socket.send_pyobj(queue[0])
                queue = queue[1:]
        elif instruction == 'ADD_RESULT':
            param_results[data[0]].append(data[1])
            socket.send_pyobj(0)
            completed_count += 1
            print('Progress = {} / {}'.format(completed_count, total_count))

            if len(param_results[data[0]]) == TEST_RUNS:
                mean = np.mean(param_results[data[0]])
                if best_result is None or mean > best_result[1]:
                    best_result = (data[0], mean)
            print(best_result)
