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
from utils import PARTICLE_SIZES
import gymenvs

import gym
import time
import torch

TASK = args.task
POLICY_MODEL_PATH = args.model_path
PARTICLE_SIZE = PARTICLE_SIZES[TASK]

if __name__ == '__main__':
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
            env.render()
            time.sleep(1) # TODO: Make this delay task independent.

            if done:
                break

