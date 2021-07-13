#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Batch Benchmarking Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--macro-length', type=int, required=True,
                    help='Macro-action length')
parser.add_argument('--num-env', type=int, default=16,
                    help='Number of environments (default: 16)')
parser.add_argument('--not-belief-dependent', dest='belief_dependent', default=True, action='store_false')
parser.add_argument('--not-context-dependent', dest='context_dependent', default=True, action='store_false')
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))

from environment import Environment, Response
from models import MAGICGenNet, MAGICCriticNet, MAGICGenNet_DriveHard, MAGICCriticNet_DriveHard
from utils import Statistics, PARTICLE_SIZES, CONTEXT_SIZES

import multiprocessing
import numpy as np
import queue
import struct
import time
import torch
import zmq
from collections import defaultdict

TASK = args.task
MACRO_LENGTH = args.macro_length
PARTICLE_SIZE = PARTICLE_SIZES[TASK] if TASK in PARTICLE_SIZES else None
CONTEXT_SIZE = CONTEXT_SIZES[TASK] if TASK in CONTEXT_SIZES else None
GAMMA = 0.98
BELIEF_DEPENDENT = args.belief_dependent
CONTEXT_DEPENDENT = args.context_dependent

NUM_ENVIRONMENTS = args.num_env
ZMQ_ADDRESS = 'tcp://127.0.0.1'

def environment_process(port):
    zmq_context = zmq.Context()
    socket = zmq_context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    environment = Environment(TASK, MACRO_LENGTH, False)
    while True:

        # Re-randomize model.
        with torch.no_grad():
            if TASK in ['DriveHard']:
                gen_model = MAGICGenNet_DriveHard(MACRO_LENGTH, BELIEF_DEPENDENT).to(device).float()
            else:
                gen_model = MAGICGenNet(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()
        #print(sum(p.numel() for p in gen_model.parameters() if p.requires_grad))

        cycles = 0
        steps = 0
        total_reward = 0
        collision = None
        macro_length = 0
        num_nodes = 0
        depth = 0

        while True:

            # Read environment state.
            context = environment.read_context()
            #context = cv2.imdecode(context, cv2.IMREAD_UNCHANGED)[...,0:2]
            state = environment.read_state()

            # Call generator if needed.
            if state is not None:
                with torch.no_grad():
                    (macro_actions) = gen_model.mode(
                            torch.tensor(context, dtype=torch.float, device=device).unsqueeze(0),
                            torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0))
                params = macro_actions.squeeze(0).cpu().numpy()
                environment.write_params(params)

            # Read response.
            response = environment.process_response()

            # Add to statistics.
            cycles += 1
            total_reward += response.undiscounted_reward
            macro_length += response.macro_length
            num_nodes = max(num_nodes, response.num_nodes)
            depth = max(depth, response.depth)
            steps += response.steps
            if response.is_terminal:
                stats = response.stats
                collision = response.is_failure
                break

        # Upload trajectory statistics.
        socket.send_pyobj((
            'ADD_TRAJECTORY_RESULT',
            steps,
            total_reward,
            collision,
            macro_length / cycles,
            num_nodes,
            depth,
            stats))
        socket.recv_pyobj()

if __name__ == '__main__':

    print('{} |{} |{} |{} |{} |{} |{} |{} |{}'.format(
        'Steps'.rjust(16, ' '),
        'Total Reward'.rjust(19, ' '),
        'Collision %'.rjust(12, ' '),
        'Num. Nodes'.rjust(19, ' '),
        'Depth'.rjust(16, ' '),
        'Stat0'.rjust(16, ' '),
        'Stat1'.rjust(16, ' '),
        'Stat2'.rjust(16, ' '),
        'Stat3'.rjust(16, ' ')))

    # Prepare zmq server.
    zmq_context = zmq.Context()
    socket = zmq_context.socket(zmq.REP)
    port = socket.bind_to_random_port(ZMQ_ADDRESS)

    # Start processes.
    processes = [multiprocessing.Process(
        target=environment_process,
        args=(port,), daemon=True) for _ in range(NUM_ENVIRONMENTS)]
    for p in processes:
        p.start()

    steps_statistics = Statistics()
    total_reward_statistics = Statistics()
    collision_statistics = Statistics()
    macro_length_statistics = Statistics()
    num_nodes_statistics = Statistics()
    depth_statistics = Statistics()
    stats_statistics = [Statistics() for _ in range(5)]
    planner_value_statistics = defaultdict(lambda: Statistics())
    value_statistics = defaultdict(lambda: Statistics())

    while True:

        # Read request and process.
        request = socket.recv_pyobj()
        instruction = request[0]
        instruction_data = request[1:]

        if instruction == 'ADD_TRAJECTORY_RESULT':

            socket.send_pyobj(0) # Return immediately.
            (steps, total_reward, collision, macro_length, num_nodes, depth, stats) = instruction_data

            if args.task not in ['DriveHard']:
                if not collision:
                    steps_statistics.append(steps)
            else:
                steps_statistics.append(steps)
            total_reward_statistics.append(total_reward)
            collision_statistics.append(collision)
            macro_length_statistics.append(macro_length)
            num_nodes_statistics.append(num_nodes)
            depth_statistics.append(depth)
            for (s, ss) in zip(stats, stats_statistics):
                if s is not None:
                    ss.extend(s)

            print('\r\033[K{} |{} |{} |{} |{} |{} |{} |{} |{}| {}'.format(
                str(steps_statistics).rjust(16, ' '),
                str(total_reward_statistics).rjust(19, ' '),
                str(collision_statistics).rjust(12, ' '),
                str(num_nodes_statistics).rjust(19, ' '),
                str(depth_statistics).rjust(16, ' '),
                str(stats_statistics[0]).rjust(16, ' '),
                str(stats_statistics[1]).rjust(16, ' '),
                str(stats_statistics[2]).rjust(16, ' '),
                str(stats_statistics[3]).rjust(16, ' '),
                str(stats_statistics[4]).rjust(16, ' ')), end="")
