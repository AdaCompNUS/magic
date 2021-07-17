#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (Handcrafted) Benchmarking Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--macro-length', type=int, required=True,
                    help='Macro-action length')
parser.add_argument('--num-env', type=int, default=16,
                    help='Number of environments (default: 16)')
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
import zmq
from collections import defaultdict

TASK = args.task
MACRO_LENGTH = args.macro_length
TARGET_SE = args.target_se
GAMMA = 0.98

NUM_ENVIRONMENTS = args.num_env
ZMQ_ADDRESS = 'tcp://127.0.0.1'

def environment_process(index, port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    environment = Environment(TASK, MACRO_LENGTH, False)

    while True:
        cycles = 0
        steps = 0
        total_reward = 0
        collision = None
        num_nodes = None
        depth = None
        while True:
            # Read step response.
            response = environment.process_response()

            # Add to statistics.
            cycles += 1
            total_reward += response.undiscounted_reward
            if response.num_nodes is not None:
                num_nodes = response.num_nodes if num_nodes is None else max(num_nodes, response.num_nodes)
            if response.depth is not None:
                depth = response.depth if depth is None else max(depth, response.depth)
            steps += response.steps
            if response.is_terminal:
                stats = response.stats
                collision = response.is_failure
                break

        socket.send_pyobj([
            steps,
            total_reward,
            collision,
            num_nodes,
            depth,
            stats])
        socket.recv_pyobj()

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
    stats_statistics = [Statistics() for _ in range(5)]
    planner_value_statistics = defaultdict(lambda: Statistics())
    value_statistics = defaultdict(lambda: Statistics())

    while len(total_reward_statistics) < 100 or total_reward_statistics.stderr() > TARGET_SE:
        # Read request and process.
        (steps, total_reward, collision, num_nodes, depth, stats) = socket.recv_pyobj()
        socket.send_pyobj(0)

        if args.task not in ['DriveHard']:
            if not collision:
                steps_statistics.append(steps)
        else:
            steps_statistics.append(steps)
        total_reward_statistics.append(total_reward)
        collision_statistics.append(collision)
        if num_nodes is not None:
            num_nodes_statistics.append(num_nodes)
        if depth is not None:
            depth_statistics.append(depth)
        for (s, ss) in zip(stats, stats_statistics):
            if s is not None:
                ss.extend(s)

        print('\r\033[K{}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}'.format(
            steps_statistics,
            total_reward_statistics,
            collision_statistics,
            num_nodes_statistics,
            depth_statistics,
            stats_statistics[0],
            stats_statistics[1],
            stats_statistics[2],
            stats_statistics[3],
            stats_statistics[4]), end="")

    for p in processes:
        p.terminate()
    print()
