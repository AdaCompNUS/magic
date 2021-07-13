#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='Macro-DESPOT (Handcrafted) Evaluation Args')
parser.add_argument('--task', required=True,
                    help='Task')
parser.add_argument('--macro-length', type=int, required=True,
                    help='Macro-action length')
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from environment import Environment, Response
import numpy as np

np.set_printoptions(precision=4, suppress=True)

TASK = args.task
MACRO_LENGTH = args.macro_length

if __name__ == '__main__':

    environment = Environment(TASK, MACRO_LENGTH, True)

    while True:
        print(environment.process_response())
