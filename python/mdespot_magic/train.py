#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Training Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--macro-length', type=int, required=True,
                    help='Macro-action length')
parser.add_argument('--num-env', type=int, default=16,
                    help='Number of environments (default: 16)')
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from environment import Environment, Response
from models import MAGICGenNet, MAGICCriticNet
from replay import ReplayBuffer
from utils import PARTICLE_SIZES

import multiprocessing
import numpy as np
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zmq
np.set_printoptions(precision=4)


TASK = args.task
MACRO_LENGTH = args.macro_length
PARTICLE_SIZE = PARTICLE_SIZES[TASK]

# Training configurations
UPDATE_INTERVAL = 1
REPLAY_MIN = 1000
REPLAY_MAX = 100000
REPLAY_SAMPLE_SIZE = 32
SAVE_INTERVAL = 1000
LR = 1e-4
PERTURBATION_STENGTH = 0.15
RECENT_HISTORY_LENGTH = 50
SAVE_PATH = 'learned_{}-{}/'.format(TASK, MACRO_LENGTH)

NUM_ENVIRONMENTS = args.num_env
ZMQ_ADDRESS = 'tcp://127.0.0.1'

def environment_process(index, port):
    print('Starting env {}...'.format(index))
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('{}:{}'.format(ZMQ_ADDRESS, port))

    environment = Environment(TASK, MACRO_LENGTH, False)
    previous_state = None
    previous_params = None
    previous_response = None
    previous_trajectory_result = None

    steps = 0
    total_reward = 0
    collision = None

    while True:
        # Read environment state, and pyobj together with previous step.
        state = environment.read_state()
        socket.send_multipart([
            struct.pack('i', index),
            memoryview(previous_state if previous_state is not None else np.array([], dtype=np.float32)),
            memoryview(previous_params if previous_params is not None else np.array([], dtype=np.float32)),
            memoryview(np.array(list(previous_response), dtype=np.float32) if previous_response is not None else np.array([], dtype=np.float32)),
            memoryview(state),
            memoryview(previous_trajectory_result if previous_trajectory_result is not None else np.array([], dtype=np.float32))])

        # Reset trajectory result.
        if previous_trajectory_result is not None:
            previous_trajectory_result = None

        # Read params.
        previous_params = np.frombuffer(socket.recv(), dtype=np.float32)

        # Apply params and save response.
        environment.write_params(previous_params)
        previous_response = environment.process_response()

        # Add to trajectory statistics
        steps += previous_response.steps
        total_reward += previous_response.reward
        if previous_response.is_terminal:
            collision = 1.0 if previous_response.reward < 0 else 0.0
            previous_trajectory_result = np.array([steps, total_reward, collision], dtype=np.float32)
            steps = 0
            total_reward = 0
            collision = None

        # Save state.
        previous_state = state

if __name__ == '__main__':

    save_path =  SAVE_PATH.format(MACRO_LENGTH)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Load models.
    print('Loading models...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen_model = MAGICGenNet(PARTICLE_SIZE).float().to(device)
    gen_model_optimizer = optim.Adam(gen_model.parameters(), lr=LR)

    critic_model = MAGICCriticNet(PARTICLE_SIZE).float().to(device)
    critic_model_optimizer = optim.Adam(critic_model.parameters(), lr=LR)

    # Prepare zmq server.
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port(ZMQ_ADDRESS)

    # Start processes.
    print('Starting processes...')
    processes = [multiprocessing.Process(target=environment_process, args=(i, port,), daemon=True) for i in range(NUM_ENVIRONMENTS)]
    for p in processes:
        p.start()

    step = 0
    start = time.time()
    recent_steps = []
    recent_total_reward = []
    recent_collisions = []
    replay_buffer = ReplayBuffer(REPLAY_MAX)

    while True:

        # Read request and process.
        request = [np.array(np.frombuffer(r, dtype=np.float32)) for r in socket.recv_multipart()]
        request = [r if r.size > 0 else None for r in request]
        (index, previous_state, previous_params, previous_response, state, previous_trajectory_result) = request
        index = struct.unpack('i', index)[0]

        if previous_state is not None:
            previous_state = previous_state.reshape((-1, PARTICLE_SIZE))
        if previous_response is not None:
            previous_response = Response(*previous_response)
        state = state.reshape((-1, PARTICLE_SIZE))

        # Respond params.
        with torch.no_grad():
            params = gen_model(torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0), PERTURBATION_STENGTH)[0].squeeze(0).float().cpu().numpy()
        socket.send(memoryview(params))

        # Add previous state, params, response to replay buffer, if any.
        if previous_response is not None:
            replay_buffer.append((previous_state, previous_params, previous_response.best_value))

        # Add last trajectory results to statistics, if any.
        if previous_trajectory_result is not None:
            recent_steps.append((previous_trajectory_result[0], previous_trajectory_result[2] > 0.5))
            recent_total_reward.append(previous_trajectory_result[1])
            recent_collisions.append(previous_trajectory_result[2])
            recent_steps = recent_steps[-RECENT_HISTORY_LENGTH:]
            recent_total_reward = recent_total_reward[-RECENT_HISTORY_LENGTH:]
            recent_collisions = recent_collisions[-RECENT_HISTORY_LENGTH:]

        if len(replay_buffer) < REPLAY_MIN:
            print('Waiting for minimum buffer size ... {}/{}'.format(len(replay_buffer), REPLAY_MIN))
            continue

        sampled_evaluations = replay_buffer.sample(REPLAY_SAMPLE_SIZE)
        sampled_states = torch.stack([torch.tensor(t[0], dtype=torch.float, device=device) for t in sampled_evaluations])
        sampled_params = torch.stack([torch.tensor(t[1], dtype=torch.float, device=device) for t in sampled_evaluations])
        sampled_values = torch.stack([torch.tensor(t[2], dtype=torch.float, device=device) for t in sampled_evaluations])

        # Update critic.
        critic_loss = torch.distributions.Normal(*critic_model(sampled_states, sampled_params)).log_prob(sampled_values).mean(dim=-1)
        critic_model_optimizer.zero_grad()
        gen_model_optimizer.zero_grad()
        (-critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 1.0)
        critic_model_optimizer.step()

        # Update params model.
        (value, sd) = critic_model(sampled_states, gen_model(sampled_states))
        params_loss = (-value).mean(dim=-1)
        critic_model_optimizer.zero_grad()
        gen_model_optimizer.zero_grad()
        params_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
        gen_model_optimizer.step()

        if previous_trajectory_result is not None:
            # Log statistics.
            print("\033[H\033[J")
            print('Step {}: {}'.format(step, previous_response))
            print('Step {}: Recent Steps (Pass) = {}'.format(step,
                np.mean([s[0] for s in recent_steps if not s[1]]) if len([s[0] for s in recent_steps if not s[1]]) > 0 else None))
            print('Step {}: Recent Steps (Fail) = {}'.format(step,
                np.mean([s[0] for s in recent_steps if s[1]]) if len([s[0] for s in recent_steps if s[1]]) > 0 else None))
            print('Step {}: Recent Total Reward = {}'.format(step, np.mean(recent_total_reward) if len(recent_total_reward) > 0 else None))
            print('Step {}: Recent Collisions = {}'.format(step, np.mean(recent_collisions) if len(recent_collisions) > 0 else None))
            print('Step {}: Critic Net Loss = {}'.format(step, critic_loss.detach().item()))
            print('Step {}: Generator Value = {}'.format(step, value.mean(dim=-1).detach().item()))
            print('Step {}: Generator Value S.D. = {}'.format(step, sd.mean(dim=-1).detach().item()))
            print('Step {}: Elapsed = {} m'.format(step, (time.time() - start) / 60))

        # Save models.
        if step % SAVE_INTERVAL == 0:
            print('Saving models....')
            torch.save(gen_model.state_dict(), save_path + 'gen_model.pt.{:08d}'.format(step))
            torch.save(critic_model.state_dict(), save_path + 'critic_model.pt.{:08d}'.format(step))

        step += 1
