#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Batch Benchmarking Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--macro-length', type=int, required=True,
                    help='Macro-action length')
parser.add_argument('--num-env', type=int, default=16,
                    help='Number of environments (default: 16)')
parser.add_argument('--models-folder', required=True,
                    help='Path to folder containing models')
parser.add_argument('--model-index', type=int, default=None,
                    help='Index of model to benchmark. Leave empty to benchmark all in folder.')
parser.add_argument('--not-belief-dependent', dest='belief_dependent', default=True, action='store_false')
parser.add_argument('--not-context-dependent', dest='context_dependent', default=True, action='store_false')
parser.add_argument('--target-se', type=float, default=None, help='Target standard error  (default: None)')
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

MODEL_PATTERN = args.models_folder + '/gen_model.pt.{:08d}'
BATCH_MODE = args.model_index is None
ITERATIONS = 1000 #DriveHard single: 2000 # DriveHard batch: 400
INTERVAL = 5000
TARGET_SE = args.target_se

def environment_process(port):
    zmq_context = zmq.Context()
    socket = zmq_context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    environment = Environment(TASK, MACRO_LENGTH, False)
    while True:

        cycles = 0
        steps = 0
        total_reward = 0
        collision = None
        macro_length = 0
        num_nodes = None
        depth = None

        while True:

            # Read environment state.
            context = environment.read_context()
            #context = cv2.imdecode(context, cv2.IMREAD_UNCHANGED)[...,0:2]
            state = environment.read_state()

            # Call generator if needed.
            if state is not None:
                socket.send_pyobj((
                    'CALL_GENERATOR',
                    context,
                    state))
                params = socket.recv_pyobj()
                environment.write_params(params)

            # Read response.
            response = environment.process_response()

            # Add to statistics.
            cycles += 1
            total_reward += response.undiscounted_reward
            macro_length += response.macro_length
            if response.num_nodes is not None:
                num_nodes = response.num_nodes if num_nodes is None else max(num_nodes, response.num_nodes)
            if response.depth is not None:
                depth = response.depth if depth is None else max(depth, response.depth)
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if TASK in ['DriveHard']:
        gen_model = MAGICGenNet_DriveHard(MACRO_LENGTH, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()
    else:
        gen_model = MAGICGenNet(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()

    print('{} |{} |{} |{} |{} |{} |{} |{} |{} |{}'.format(
        'Batch'.rjust(10, ' '),
        'Steps'.rjust(16, ' '),
        'Total Reward'.rjust(19, ' '),
        'Collision %'.rjust(12, ' '),
        #'Macro. Length'.rjust(14, ' '),
        'Num. Nodes'.rjust(19, ' '),
        'Depth'.rjust(16, ' '),
        'Stat0'.rjust(16, ' '),
        'Stat1'.rjust(16, ' '),
        'Stat2'.rjust(16, ' '),
        'Stat3'.rjust(16, ' ')))

    if BATCH_MODE:
        batch_step = 0

    while True:
        if not BATCH_MODE:
            model_path = MODEL_PATTERN.format(args.model_index)
        else:
            if not os.path.exists(MODEL_PATTERN.format(batch_step)):
                break
            model_path = MODEL_PATTERN.format(batch_step)

        # Prepare zmq server.
        zmq_context = zmq.Context()
        socket = zmq_context.socket(zmq.REP)
        port = socket.bind_to_random_port(ZMQ_ADDRESS)

        # Start processes.
        processes = [multiprocessing.Process(
            target=environment_process,
            args=(port,), daemon=True) for i in range(NUM_ENVIRONMENTS)]
        for p in processes:
            p.start()

        # Load model.
        gen_model.load_state_dict(torch.load(model_path))

        steps_statistics = Statistics()
        total_reward_statistics = Statistics()
        collision_statistics = Statistics()
        macro_length_statistics = Statistics()
        num_nodes_statistics = Statistics()
        depth_statistics = Statistics()
        stats_statistics = [Statistics() for _ in range(5)]
        planner_value_statistics = defaultdict(lambda: Statistics())
        value_statistics = defaultdict(lambda: Statistics())

        completed_iterations = 0
        while completed_iterations < ITERATIONS or (TARGET_SE is not None and total_reward_statistics.stderr() > TARGET_SE):

            # Read request and process.
            request = socket.recv_pyobj()
            instruction = request[0]
            instruction_data = request[1:]

            if instruction == 'CALL_GENERATOR':

                context = instruction_data[0]
                state = instruction_data[1]
                with torch.no_grad():
                    (macro_actions) = gen_model.mode(
                            torch.tensor(instruction_data[0], dtype=torch.float, device=device).unsqueeze(0),
                            torch.tensor(instruction_data[1], dtype=torch.float, device=device).unsqueeze(0))
                params = macro_actions.squeeze(0).cpu().numpy()
                socket.send_pyobj(params)

            elif instruction == 'ADD_TRAJECTORY_RESULT':

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
                if num_nodes is not None:
                    num_nodes_statistics.append(num_nodes)
                if depth is not None:
                    depth_statistics.append(depth)
                for (s, ss) in zip(stats, stats_statistics):
                    if s is not None:
                        ss.extend(s)
                completed_iterations += 1

                print('\r\033[K{} |{} |{} |{} |{} |{} |{} |{} |{} |{}| {}'.format(
                    str(batch_step if BATCH_MODE else args.model_index).rjust(10, ' '),
                    str(steps_statistics).rjust(16, ' '),
                    str(total_reward_statistics).rjust(19, ' '),
                    str(collision_statistics).rjust(12, ' '),
                    #str(macro_length_statistics).rjust(14, ' '),
                    str(num_nodes_statistics).rjust(19, ' '),
                    str(depth_statistics).rjust(16, ' '),
                    str(stats_statistics[0]).rjust(16, ' '),
                    str(stats_statistics[1]).rjust(16, ' '),
                    str(stats_statistics[2]).rjust(16, ' '),
                    str(stats_statistics[3]).rjust(16, ' '),
                    str(stats_statistics[4]).rjust(16, ' ')), end="")


        # Destroy processes.
        for p in processes:
            p.terminate()
            p.join()
        socket.close()

        '''
        print('{} |{} |{} |{} |{} |{} |{} |{} |{} |{}| {}'.format(
            str(batch_step if BATCH_MODE else args.model_index).rjust(10, ' '),
            str(steps_statistics).rjust(16, ' '),
            str(total_reward_statistics).rjust(19, ' '),
            str(collision_statistics).rjust(12, ' '),
            #str(macro_length_statistics).rjust(14, ' '),
            str(num_nodes_statistics).rjust(19, ' '),
            str(depth_statistics).rjust(16, ' '),
            str(stats_statistics[0]).rjust(16, ' '),
            str(stats_statistics[1]).rjust(16, ' '),
            str(stats_statistics[2]).rjust(16, ' '),
            str(stats_statistics[3]).rjust(16, ' '),
            str(stats_statistics[4]).rjust(16, ' ')))
        '''
        print()

        if not BATCH_MODE:
            break
        else:
            batch_step += INTERVAL
