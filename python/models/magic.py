#!/usr/bin/env python3

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MACRO_CURVE_ORDER = 3
NUM_CURVES = 8
EPS_SOFTPLUS = 0.01

class MAGICParamsNet(nn.Module):
    def __init__(self, particle_size):

        super(MAGICParamsNet, self).__init__()
        self.particle_size = particle_size

        self.fc1 = nn.Linear(particle_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 512)

        # Output layer.
        self.fc10 = nn.Linear(512, NUM_CURVES * (2 * MACRO_CURVE_ORDER))

    def forward(self, x, perturbation_strength=0):

        # FC for each particle.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.mean(dim=-2)

        # Mean latent variables -> latent dstribution parameters.
        x_skip = x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x_skip = x = F.relu(self.fc5(x) + x_skip)
        x = self.fc6(x)
        x_skip = x = F.relu(self.fc7(x) + x_skip)
        x = self.fc8(x)
        x_skip = x = F.relu(self.fc9(x) + x_skip)
        x = self.fc10(x)

        # Normalize each macro-action.
        x = x.view((-1, NUM_CURVES, MACRO_CURVE_ORDER * 2))
        x = x / (x**2).sum(dim=-1, keepdim=True).sqrt()
        if perturbation_strength > 0:
            perturbation = torch.randn(x.shape, device=x.device, dtype=x.dtype)
            perturbation = perturbation / (perturbation**2).sum(dim=-1, keepdim=True).sqrt()
            x = x + perturbation_strength * perturbation;
            x = x / (x**2).sum(dim=-1, keepdim=True).sqrt()
        x = x.view((-1, NUM_CURVES * (2 * MACRO_CURVE_ORDER)))

        return x

class MAGICCriticNet(nn.Module):
    def __init__(self, particle_size):
        super(MAGICCriticNet, self).__init__()
        self.particle_size = particle_size

        self.fc1 = nn.Linear(particle_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(256 + NUM_CURVES * 2 * MACRO_CURVE_ORDER, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 512)
        self.fc10 = nn.Linear(512, 2)

    def forward(self, x, action):

        # FC for each particle.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.mean(dim=-2)

        x = torch.cat([x, action], dim=-1)
        x_skip = x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x_skip = x = F.relu(self.fc5(x) + x_skip)
        x = F.relu(self.fc6(x))
        x_skip = x = F.relu(self.fc7(x) + x_skip)
        x = F.relu(self.fc8(x))
        x_skip = x = F.relu(self.fc9(x) + x_skip)
        x = self.fc10(x)

        return (x[...,0], EPS_SOFTPLUS + F.softplus(x[...,1]))
