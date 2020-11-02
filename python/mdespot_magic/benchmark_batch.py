#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Batch Benchmarking Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--macro-length', type=int, required=True,
                    help='Macro-action length')
parser.add_argument('--num-env', type=int, default=8,
                    help='Number of environments (default: 8)')
parser.add_argument('--models-folder', required=True,
                    help='Path to folder containing models')
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))

from environment import Environment, Response
from models import MAGICParamsNet, MAGICCriticNet
from utils import Statistics, PARTICLE_SIZES

import multiprocessing
import numpy as np
import queue
import struct
import time
import torch
import zmq

TASK = args.task
MACRO_LENGTH = args.macro_length
PARTICLE_SIZE = PARTICLE_SIZES[TASK]

NUM_ENVIRONMENTS = args.num_env
ZMQ_ADDRESS = 'tcp://127.0.0.1'

BATCH_PATTERN = args.models_folder + '/params_model.pt.{:08d}'
ITERATIONS = 400
INTERVAL = 4000

def environment_process(port, remaining, lock):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    environment = Environment(TASK, MACRO_LENGTH, False)
    while True:
        lock.acquire()
        r = remaining.value
        if r >= 1:
            remaining.value -= 1
        lock.release()
        if r == 0:
            break

        steps = 0
        total_reward = 0
        collision = None
        num_nodes = 0
        depth = 0
        tracking_error = None
        while True:
            # Read environment state.
            state = environment.read_state()
            socket.send_multipart([memoryview(state)])

            # Read params.
            params = np.frombuffer(socket.recv(), dtype=np.float32)

            # Apply params and save response.
            environment.write_params(params)
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

    batch_step = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params_model = MAGICParamsNet(PARTICLE_SIZE).to(device).float()

    while True:
        if not os.path.exists(BATCH_PATTERN.format(batch_step)):
            break

        # Prepare zmq server.
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        port = socket.bind_to_random_port(ZMQ_ADDRESS)

        # Start processes.
        remaining = multiprocessing.Value('i', ITERATIONS, lock=False)
        lock = multiprocessing.RLock()
        processes = [multiprocessing.Process(
            target=environment_process,
            args=(port, remaining, lock), daemon=True) for i in range(NUM_ENVIRONMENTS)]
        for p in processes:
            p.start()

        # Load model.
        params_model.load_state_dict(torch.load(BATCH_PATTERN.format(batch_step)))

        steps_statistics = Statistics()
        total_reward_statistics = Statistics()
        collision_statistics = Statistics()
        num_nodes_statistics = Statistics()
        depth_statistics = Statistics()
        tracking_error_statistics = Statistics()

        completed_iterations = 0
        while completed_iterations < ITERATIONS:

            # Read request and process.
            data = socket.recv_multipart()
            if len(data) > 1:
                (steps, total_reward, collision, num_nodes, depth) = (struct.unpack(f, d)[0] for (f, d) in zip(['i', 'f', 'i', 'i', 'i'], data))
                if collision < 0.5:
                    steps_statistics.append(struct.unpack('i', data[0])[0])
                total_reward_statistics.append(struct.unpack('f', data[1])[0])
                collision_statistics.append(struct.unpack('i', data[2])[0])
                num_nodes_statistics.append(struct.unpack('i', data[3])[0])
                depth_statistics.append(struct.unpack('i', data[4])[0])
                tracking_error_statistics.append(struct.unpack('f', data[5])[0])
                socket.send_multipart([b''])
                completed_iterations += 1
            else:
                state = np.frombuffer(data[0], dtype=np.float32).reshape((-1, PARTICLE_SIZE))
                with torch.no_grad():
                    params = params_model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))[0].squeeze(0).cpu().numpy()
                    socket.send(memoryview(params))

        # Destroy processes.
        for p in processes:
            p.terminate()
            p.join()
        socket.close()

        print('{:08d}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}'.format(
            batch_step,
            steps_statistics,
            total_reward_statistics,
            collision_statistics,
            num_nodes_statistics,
            depth_statistics,
            tracking_error_statistics))
        batch_step += INTERVAL
