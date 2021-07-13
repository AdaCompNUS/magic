#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()

import torch
import json
import os
import sys
sys.path.append('{}/../'.format(os.path.dirname(os.path.realpath(__file__))))
from models import MAGICGenNet, MAGICCriticNet

import socket
import ssl

BATCH_SIZE = 7000
NUM_PARTICLES = 100

if __name__ == '__main__':

    # Read config.
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    save_url = cfg['save-url']
    save_port = cfg['save-port']
    cert = cfg['cert']

    # Prepare models.
    devices = [torch.device("cuda:{}".format(i) if torch.cuda.is_available() else "cpu") for i in range(4)]
    critic_models = [MAGICCriticNet(3, 3, True, True).float().to(device) for device in devices]
    for critic_model in critic_models:
        critic_model.load_state_dict(torch.load(cfg['model']))

    # Prepare write to remote server.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
        with context.wrap_socket(sock) as ssock:
            ssock.connect((save_url, save_port))

            while True:
                for (device, critic_model) in zip(devices, critic_models):
                    try:
                        c = torch.rand((BATCH_SIZE, 3), device=device)
                        x = torch.rand((BATCH_SIZE, NUM_PARTICLES, 3), device=device)
                        a = torch.rand((BATCH_SIZE, 8 * 2 * 3), device=device)
                        critic_model(c, x, a)[0].mean().backward()
                        ssock.send(b'BOMB PASSED!')
                    except:
                        ssock.send(b'BOMB FAILED!')

                    if ssock.recv(1024) != b'ACK':
                        raise Exception('CONNECTION ERROR!')
