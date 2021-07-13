#!/usr/bin/env python3

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from power_spherical import PowerSpherical

MACRO_CURVE_ORDER = 3
NUM_CURVES = 8
EPS = 0.01

def LeakyHardTanh(x, x_min=-1, x_max=1, hard_slope=1e-2):
    return (x >= x_min) * (x <= x_max) * x + (x < x_min) * (x_min + hard_slope * (x - x_min)) + (x > x_max) * (x_max + hard_slope * (x - x_max))

def LeakyReLUTop(x, x_max=1, hard_slope=1e-2):
    return (x <= x_max) * x + (x > x_max) * (x_max + hard_slope * (x - x_max))

class MAGICGenNet(nn.Module):
    def __init__(self, context_size, particle_size, context_dependent, belief_dependent):

        super(MAGICGenNet, self).__init__()
        self.context_size = context_size
        self.particle_size = particle_size
        self.context_dependent = context_dependent
        self.belief_dependent = belief_dependent

        if self.context_dependent or self.belief_dependent:

            if self.belief_dependent:
                self.fc1 = nn.Linear(particle_size, 256)
                self.fc2 = nn.Linear(256, 256)

            self.fc3 = nn.Linear(context_size * self.context_dependent + 256 * self.belief_dependent, 512)
            self.fc4 = nn.Linear(512, 512)
            self.fc5 = nn.Linear(512, 512)
            self.fc6 = nn.Linear(512, 512)
            self.fc7 = nn.Linear(512, 512)
            self.fc8 = nn.Linear(512, 512)
            self.fc9 = nn.Linear(512, 512)

            # Output layer.
            self.fc10_mean = nn.Linear(512, NUM_CURVES * (2 * MACRO_CURVE_ORDER))
            self.fc10_concentration = nn.Linear(512, NUM_CURVES)
            torch.nn.init.constant_(self.fc10_concentration.bias, 100)

        else:
            self.mean = nn.Parameter(torch.normal(torch.zeros(NUM_CURVES * (2 * MACRO_CURVE_ORDER)), 1), requires_grad=True)
            self.concentration = nn.Parameter(100 * torch.ones(NUM_CURVES), requires_grad=True)

    def forward(self, c, x):

        x = x.reshape((x.shape[0], -1, self.particle_size))

        if self.context_dependent or self.belief_dependent:

            if self.belief_dependent:
                # FC for each particle.
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = x.mean(dim=-2)

            x = torch.cat(
                    ([c] if self.context_dependent else []) \
                    + ([x] if self.belief_dependent else []), dim=-1)

            # Mean latent variables -> latent dstribution parameters.
            x_skip = x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x_skip = x = F.relu(self.fc5(x) + x_skip)
            x = F.relu(self.fc6(x))
            x_skip = x = F.relu(self.fc7(x) + x_skip)
            x = F.relu(self.fc8(x))
            x_skip = x = F.relu(self.fc9(x) + x_skip)

            mean = self.fc10_mean(x)
            mean = mean.view((-1, NUM_CURVES, 2 * MACRO_CURVE_ORDER))
            mean = mean / (mean**2).sum(dim=-1, keepdim=True).sqrt()

            concentration = 1 + F.softplus(self.fc10_concentration(x))

            return (PowerSpherical(mean, concentration),)

        else:
            mean = torch.cat(x.shape[0] * [self.mean.unsqueeze(0)], dim=0)
            concentration = torch.cat(x.shape[0] * [self.concentration.unsqueeze(0)], dim=0)

            mean = mean.view((-1, NUM_CURVES, 2 * MACRO_CURVE_ORDER))
            mean = mean / (mean**2).sum(dim=-1, keepdim=True).sqrt()
            concentration = 1 + F.softplus(concentration)

            return (PowerSpherical(mean, concentration),)

    def rsample(self, c, x):
        (macro_actions_dist,) = self.forward(c, x)
        macro_actions = macro_actions_dist.rsample()
        macro_actions = macro_actions.view((-1, NUM_CURVES * 2 * MACRO_CURVE_ORDER))
        macro_actions_entropy = macro_actions_dist.entropy()

        return (macro_actions, macro_actions_entropy)

    def mode(self, c, x):
        (macro_actions_dist,) = self.forward(c, x)
        macro_actions = macro_actions_dist.loc.view((-1, NUM_CURVES * 2 * MACRO_CURVE_ORDER))

        return macro_actions

class MAGICCriticNet(nn.Module):
    def __init__(self, context_size, particle_size, context_dependent, belief_dependent):
        super(MAGICCriticNet, self).__init__()
        self.context_size = context_size
        self.particle_size = particle_size
        self.context_dependent = context_dependent
        self.belief_dependent = belief_dependent

        if self.belief_dependent:
            self.fc1 = nn.Linear(particle_size, 256)
            self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(
                context_size * self.context_dependent \
                + 256 * self.belief_dependent \
                + NUM_CURVES * 2 * MACRO_CURVE_ORDER, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 512)
        self.fc10 = nn.Linear(512, 2)
        self.fc10.bias.data[1] = 100

    def forward(self, c, x, action):

        x = x.reshape((x.shape[0], -1, self.particle_size))

        if self.belief_dependent:
            # FC for each particle.
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = x.mean(dim=-2)

        x = torch.cat(
                ([c] if self.context_dependent else []) \
                + ([x] if self.belief_dependent else []) \
                + [action], dim=-1)

        x_skip = x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x_skip = x = F.relu(self.fc5(x) + x_skip)
        x = F.relu(self.fc6(x))
        x_skip = x = F.relu(self.fc7(x) + x_skip)
        x = F.relu(self.fc8(x))
        x_skip = x = F.relu(self.fc9(x) + x_skip)
        x = self.fc10(x)

        return (x[...,0], EPS + F.softplus(x[...,1]))

class MAGICGenNet_DriveHard(nn.Module):
    def __init__(self, macro_length, context_dependent, belief_dependent):

        super(MAGICGenNet_DriveHard, self).__init__()
        self.context_size = 300
        self.num_exo_agents = 15
        self.ego_size = 6 + 4 + 1
        self.exo_size = 6 + 4
        self.context_dependent = context_dependent
        self.belief_dependent = belief_dependent

        if self.context_dependent or self.belief_dependent:

            if self.belief_dependent:
                self.exo_fc1 = nn.Linear(self.exo_size, 64)
                self.exo_fc2 = nn.Linear(64, 64)
                self.exo_fc3 = nn.Linear(64, 64)

                self.particle_fc1 = nn.Linear(self.ego_size + 64, 128)
                #self.particle_fc1 = nn.Linear(self.ego_size, 128)
                self.particle_fc2 = nn.Linear(128, 128)
                self.particle_fc3 = nn.Linear(128, 128)

            # FC over combined context and belief.
            self.fc1 = nn.Linear(self.context_size * self.context_dependent + 128 * self.belief_dependent, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 512)
            self.fc4 = nn.Linear(512, 512)
            self.fc5 = nn.Linear(512, 512)
            self.fc6 = nn.Linear(512, 512)
            self.fc7 = nn.Linear(512, 512)

            # Output layer.
            self.fc8_mean = nn.Linear(512, 14)
            self.fc8_std = nn.Linear(512, 14)
            torch.nn.init.constant_(self.fc8_std.bias, 1)
        else:
            self.mean = nn.Parameter(torch.normal(torch.zeros(14), 1), requires_grad=True)
            self.std = nn.Parameter(1 * torch.ones(14), requires_grad=True)

    def forward(self, c, x):

        if self.context_dependent or self.belief_dependent:
            if self.belief_dependent:
                x = x.reshape((x.shape[0], -1, self.ego_size + self.num_exo_agents * self.exo_size))

                x_ego = x[...,:self.ego_size]

                x_exo = x[...,self.ego_size:].reshape((x.shape[0], -1, self.num_exo_agents, self.exo_size))
                x_exo = F.relu(self.exo_fc1(x_exo))
                x_exo = F.relu(self.exo_fc2(x_exo))
                x_exo = F.relu(self.exo_fc3(x_exo))
                x_exo = x_exo.mean(dim=-2) # Merge across exo agents.

                x_particle = torch.cat([x_ego, x_exo], dim=-1)
                x_particle = F.relu(self.particle_fc1(x_particle))
                x_particle = F.relu(self.particle_fc2(x_particle))
                x_particle = F.relu(self.particle_fc3(x_particle))
                x = x_particle.mean(dim=-2) # Merge across particles.


            x = torch.cat(
                    ([c] if self.context_dependent else []) \
                    + ([x] if self.belief_dependent else []), dim=-1)

            x_skip = x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x_skip = x = F.relu(self.fc3(x + x_skip))
            x = F.relu(self.fc4(x))
            x_skip = x = F.relu(self.fc5(x + x_skip))
            x = F.relu(self.fc6(x))
            x_skip = x = F.relu(self.fc7(x + x_skip))

            mean = LeakyHardTanh(self.fc8_mean(x), -5, 5)
            std = LeakyReLUTop(F.softplus(self.fc8_std(x)) + EPS, 5)

            return distributions.Normal(mean, std)
        else:
            mean = LeakyHardTanh(torch.cat(x.shape[0] * [self.mean.unsqueeze(0)], dim=0), -5, 5)
            std = torch.cat(x.shape[0] * [self.std.unsqueeze(0)], dim=0)
            std = LeakyReLUTop(F.softplus(std) + EPS, 5)

            return distributions.Normal(mean, std)

    def rsample(self, c, x):
        macro_actions_dist = self.forward(c, x)
        macro_actions_x_t = macro_actions_dist.rsample()
        macro_actions_y_t = torch.tanh(macro_actions_x_t)
        macro_actions_entropy = -macro_actions_dist.log_prob(macro_actions_x_t)
        macro_actions_entropy += torch.log(1 - macro_actions_y_t.pow(2) + 1e-6)

        return (macro_actions_y_t, macro_actions_entropy)

    def mode(self, c, x):
        macro_actions_dist = self.forward(c, x)
        macro_actions = torch.tanh(macro_actions_dist.mean)

        return macro_actions

class MAGICCriticNet_DriveHard(nn.Module):

    def __init__(self, macro_length, context_dependent, belief_dependent):
        super(MAGICCriticNet_DriveHard, self).__init__()
        self.context_size = 300
        self.num_exo_agents = 15
        self.ego_size = 6 + 4 + 1
        self.exo_size = 6 + 4
        self.context_dependent = context_dependent
        self.belief_dependent = belief_dependent

        if self.belief_dependent:
            self.exo_fc1 = nn.Linear(self.exo_size, 64)
            self.exo_fc2 = nn.Linear(64, 64)
            self.exo_fc3 = nn.Linear(64, 64)


            self.particle_fc1 = nn.Linear(self.ego_size + 64, 128)
            self.particle_fc2 = nn.Linear(128, 128)
            self.particle_fc3 = nn.Linear(128, 128)

        self.fc1 = nn.Linear(
                self.context_size * self.context_dependent \
                + 128 * self.belief_dependent \
                + 14, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 2)
        self.fc8.bias.data[1] = 100

    def forward(self, c, x, action):

        x = x.reshape((x.shape[0], -1, self.ego_size + self.num_exo_agents * self.exo_size))

        if self.belief_dependent:
            x_ego = x[...,:self.ego_size]

            x_exo = x[...,self.ego_size:].reshape((x.shape[0], -1, self.num_exo_agents, self.exo_size))
            x_exo = F.relu(self.exo_fc1(x_exo))
            x_exo = F.relu(self.exo_fc2(x_exo))
            x_exo = F.relu(self.exo_fc3(x_exo))
            x_exo = x_exo.mean(dim=-2) # Merge across exo agents.

            x_particle = torch.cat([x_ego, x_exo], dim=-1)
            x_particle = F.relu(self.particle_fc1(x_particle))
            x_particle = F.relu(self.particle_fc2(x_particle))
            x_particle = F.relu(self.particle_fc3(x_particle))
            x = x_particle.mean(dim=-2) # Merge across particles.

        x = torch.cat(
                ([c] if self.context_dependent else []) \
                + ([x] if self.belief_dependent else []) \
                + [action], dim=-1)

        x_skip = x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_skip = x = F.relu(self.fc3(x + x_skip))
        x = F.relu(self.fc4(x))
        x_skip = x = F.relu(self.fc5(x + x_skip))
        x = F.relu(self.fc6(x))
        x_skip = x = F.relu(self.fc7(x + x_skip))

        x = self.fc8(x)

        return (x[...,0], EPS + F.softplus(x[...,1]))

