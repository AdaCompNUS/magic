#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (MAGIC) Evaluation Args')
parser.add_argument('--task', required=True, help='Task')
parser.add_argument('--macro-length', type=int, required=True, help='Macro-action length')
parser.add_argument('--model-path', required=False, help='Path to model file')
parser.add_argument('--not-belief-dependent', dest='belief_dependent', default=True, action='store_false')
parser.add_argument('--not-context-dependent', dest='context_dependent', default=True, action='store_false')
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from environment import Environment, Response
from models import MAGICGenNet, MAGICCriticNet, MAGICGenNet_DriveHard, MAGICCriticNet_DriveHard
from utils import Statistics, PARTICLE_SIZES, CONTEXT_SIZES

import torch
import numpy as np
import cv2
np.set_printoptions(precision=4, suppress=True)

TASK = args.task
MACRO_LENGTH = args.macro_length
PARTICLE_SIZE = PARTICLE_SIZES[TASK] if TASK in PARTICLE_SIZES else None
CONTEXT_SIZE = CONTEXT_SIZES[TASK] if TASK in CONTEXT_SIZES else None
BELIEF_DEPENDENT = args.belief_dependent
CONTEXT_DEPENDENT = args.context_dependent

if __name__ == '__main__':
    model_path = args.model_path

    print('Loading model... ({})'.format(model_path))
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if TASK in ['DriveHard']:
            gen_model = MAGICGenNet_DriveHard(MACRO_LENGTH, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()
        else:
            gen_model = MAGICGenNet(CONTEXT_SIZE, PARTICLE_SIZE, CONTEXT_DEPENDENT, BELIEF_DEPENDENT).to(device).float()
        if model_path is not None:
            gen_model.load_state_dict(torch.load(model_path))
        gen_model.eval()

        env = Environment(TASK, MACRO_LENGTH, True)

        while True:

            context = env.read_context()
            print(context)
            #context = cv2.imdecode(context, cv2.IMREAD_UNCHANGED)[...,0:2]

            state = env.read_state()
            if state is not None:
                (macro_actions) = gen_model.mode(
                        torch.tensor(context, dtype=torch.float, device=device).unsqueeze(0),
                        torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0))
                params = macro_actions.squeeze(0).cpu().numpy()
                print(params)
                env.write_params(params)
            response = env.process_response()
