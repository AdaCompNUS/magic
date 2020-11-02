import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import numpy as np
from power_spherical import PowerSpherical

class RLCriticNet(nn.Module):
    def __init__(self, particle_size, has_bool):
        super(RLCriticNet, self).__init__()

        self.particle_size = particle_size
        self.has_bool = has_bool

        self.fc1_1 = nn.Linear(particle_size, 256)
        self.fc2_1 = nn.Linear(256, 256)
        self.fc3_1 = nn.Linear(256 + 2 + int(has_bool), 256)
        self.fc4_1 = nn.Linear(256, 256)
        self.fc5_1 = nn.Linear(256, 1)

        self.fc1_2 = nn.Linear(particle_size, 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.fc3_2 = nn.Linear(256 + 2 + int(has_bool), 256)
        self.fc4_2 = nn.Linear(256, 256)
        self.fc5_2 = nn.Linear(256, 1)

    def forward(self, x, action):

        # FC for each particle.
        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc2_1(x1))
        x1 = x1.mean(dim=-2)
        x1 = torch.cat([x1, action], dim=-1)
        x1 = F.relu(self.fc3_1(x1))
        x1 = F.relu(self.fc4_1(x1))
        x1 = self.fc5_1(x1)

        x2 = F.relu(self.fc1_2(x))
        x2 = F.relu(self.fc2_2(x2))
        x2 = x2.mean(dim=-2)
        x2 = torch.cat([x2, action], dim=-1)
        x2 = F.relu(self.fc3_2(x2))
        x2 = F.relu(self.fc4_2(x2))
        x2 = self.fc5_2(x2)

        return (x1, x2)

class RLActorNet(nn.Module):
    def __init__(self, particle_size, has_bool):
        super(RLActorNet, self).__init__()
        self.particle_size = particle_size
        self.has_bool = has_bool

        self.fc1 = nn.Linear(self.particle_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 3 + 2 * int(has_bool))

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.mean(dim=-2)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x

    def sample(self, x):

        x = self.forward(x)
        mean = x[...,:2] / (x[...,:2]**2).sum(dim=-1, keepdim=True).sqrt()
        std = 1 + F.softplus(x[...,2])
        dist = PowerSpherical(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        if self.has_bool:
            x_dist_bool = torch.distributions.Normal(x[...,3], 1e-3 + F.softplus(x[...,4]))
            x_t_bool = x_dist_bool.rsample()
            y_t_bool = torch.tanh(x_t_bool)
            action_bool = y_t_bool * 0.5 + 0.5
            y_t_bool_log_prob = x_dist_bool.log_prob(x_t_bool)
            y_t_bool_log_prob -= torch.log(0.5 * (1 - y_t_bool.pow(2)) + 1e-6)

            action = torch.cat([action, action_bool.unsqueeze(-1)], dim=-1)
            log_prob += y_t_bool_log_prob
            mean_bool = torch.tanh(x[...,3]) * 0.5 + 0.5
            mean = torch.cat([action, mean_bool.unsqueeze(-1)], dim=-1)

        return (action, log_prob.unsqueeze(-1), mean)

