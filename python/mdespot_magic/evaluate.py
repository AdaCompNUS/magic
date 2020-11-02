#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Evaluation Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--macro-length', type=int, required=True,
                    help='Macro-action length')
parser.add_argument('--model-path', required=True,
                    help='Path to model file')
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from environment import Environment, Response
from models import MAGICGenNet, MAGICCriticNet
from utils import Statistics, PARTICLE_SIZES

import torch
import numpy as np

TASK = args.task
MACRO_LENGTH = args.macro_length
PARTICLE_SIZE = PARTICLE_SIZES[TASK]

if __name__ == '__main__':
    model_path = args.model_path

    print('Loading model... ({})'.format(model_path))
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gen_model = MAGICGenNet(PARTICLE_SIZE).to(device).float()
        if model_path is not None:
            gen_model.load_state_dict(torch.load(model_path))
        gen_model.eval()

        env = Environment(TASK, MACRO_LENGTH, True)

        while True:
            state = env.read_state()
            params = gen_model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))[0].squeeze(0).cpu().numpy()
            print(params)
            env.write_params(params)
            response = env.process_response()
            print(response)
