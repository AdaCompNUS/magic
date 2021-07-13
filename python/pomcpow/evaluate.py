#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser(description='POMCPOW Evaluation Args')
parser.add_argument('--task', required=True,
                    help='Task')
args = parser.parse_args()

import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from environment import Environment, Response

TASK = args.task

if __name__ == '__main__':

    environment = Environment(TASK, True)

    while True:
        print(environment.process_response())
