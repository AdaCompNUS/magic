#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (Handcrafted) Benchmarking Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--macro-length', type=int, required=True,
                    help='Macro-action length')
parser.add_argument('--num-env', type=int, default=8,
                    help='Number of environments (default: 8)')
parser.add_argument('--target-se', type=float, default=0.1,
                    help='Target standard error  (default: 0.1)')
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from environment import Environment, Response
from utils import Statistics

import multiprocessing
import numpy as np
import queue
import struct
import time
import torch
import zmq

TASK = args.task
MACRO_LENGTH = args.macro_length
TARGET_SE = args.target_se

NUM_ENVIRONMENTS = args.num_env
ZMQ_ADDRESS = 'tcp://127.0.0.1'

def environment_process(index, port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    environment = Environment(TASK, MACRO_LENGTH, False)

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

if __name__ == '__main__':


    # Prepare zmq server.
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port(ZMQ_ADDRESS)

    # Start processes.
    processes = [multiprocessing.Process(target=environment_process, args=(i, port,), daemon=True) for i in range(NUM_ENVIRONMENTS)]
    for p in processes:
        p.start()

    steps_statistics = Statistics()
    total_reward_statistics = Statistics()
    collision_statistics = Statistics()
    num_nodes_statistics = Statistics()
    depth_statistics = Statistics()
    tracking_error_statistics = Statistics()

    print('Steps\t\t\t| Reward\t\t| Failure %\t\t| # Nodes\t\t| Search Depth\t\t| Tracking Error')
    while len(total_reward_statistics) < 1000 or total_reward_statistics.stderr() > TARGET_SE:
        # Read request and process.
        data = socket.recv_multipart()
        (steps, total_reward, collision, num_nodes, depth, tracking_error) = \
                (struct.unpack(f, d)[0] for (f, d) in zip(['i', 'f', 'i', 'i', 'i', 'f'], data))

        if collision < 0.5:
            steps_statistics.append(steps)
        total_reward_statistics.append(total_reward)
        collision_statistics.append(collision)
        num_nodes_statistics.append(num_nodes)
        depth_statistics.append(depth)
        tracking_error_statistics.append(tracking_error)
        socket.send_multipart([b''])
        print('\r\033[K{}\t| {}\t| {}\t| {}\t| {}\t| {}'.format(
            steps_statistics,
            total_reward_statistics,
            collision_statistics,
            num_nodes_statistics,
            depth_statistics,
            tracking_error_statistics), end="")

    for p in processes:
        p.terminate()
    print()
