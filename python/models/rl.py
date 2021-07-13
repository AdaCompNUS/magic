import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
from power_spherical import PowerSpherical

class RLCriticNet(nn.Module):
    def __init__(self, context_size, particle_size, has_bool):
        super(RLCriticNet, self).__init__()

        self.context_size = context_size
        self.particle_size = particle_size
        self.has_bool = has_bool

        self.fc1_1 = nn.Linear(particle_size, 256)
        self.fc2_1 = nn.Linear(256, 256)
        self.fc3_1 = nn.Linear(context_size + 256 + 2 + 2 * int(has_bool), 256)
        self.fc4_1 = nn.Linear(256, 256)
        self.fc5_1 = nn.Linear(256, 1)

        self.fc1_2 = nn.Linear(particle_size, 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.fc3_2 = nn.Linear(context_size + 256 + 2 + 2 * int(has_bool), 256)
        self.fc4_2 = nn.Linear(256, 256)
        self.fc5_2 = nn.Linear(256, 1)

    def forward(self, x, action):

        (c, x) = x

        # FC for each particle.
        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc2_1(x1))
        x1 = x1.mean(dim=-2)
        x1 = torch.cat([c, x1, action], dim=-1)
        x1 = F.relu(self.fc3_1(x1))
        x1 = F.relu(self.fc4_1(x1))
        x1 = self.fc5_1(x1)

        x2 = F.relu(self.fc1_2(x))
        x2 = F.relu(self.fc2_2(x2))
        x2 = x2.mean(dim=-2)
        x2 = torch.cat([c, x2, action], dim=-1)
        x2 = F.relu(self.fc3_2(x2))
        x2 = F.relu(self.fc4_2(x2))
        x2 = self.fc5_2(x2)

        return (x1, x2)

class RLActorNet(nn.Module):
    def __init__(self, context_size, particle_size, has_bool):
        super(RLActorNet, self).__init__()

        self.context_size = context_size
        self.particle_size = particle_size
        self.has_bool = has_bool

        self.fc1 = nn.Linear(self.particle_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(context_size + 256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5_action_mean = nn.Linear(256, 2)
        self.fc5_action_std = nn.Linear(256, 1)
        torch.nn.init.constant_(self.fc5_action_std.bias, 100)

        if has_bool:
            self.fc5_bool = nn.Linear(256, 2)

    def forward(self, x):

        (c, x) = x

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.mean(dim=-2)
        x = torch.cat([c, x], dim=-1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x_action_mean = self.fc5_action_mean(x)
        x_action_std = (1 + F.softplus(self.fc5_action_std(x))).squeeze(-1)

        mean = x_action_mean / (x_action_mean**2).sum(dim=-1, keepdim=True).sqrt()
        concentration = 1 + F.softplus(x_action_std)
        x_action_dist = PowerSpherical(mean, concentration)

        if not self.has_bool:
            return x_action_dist
        else:
            x_bool = self.fc5_bool(x)
            x_bool_dist = distributions.Categorical(logits=x_bool)
            return (x_action_dist, x_bool_dist)

    def sample(self, x):

        x = self.forward(x)

        if not self.has_bool:
            action = x.rsample()
            log_prob = x.log_prob(action).unsqueeze(-1)
        else:
            action = x[0].rsample()
            log_prob = x[0].log_prob(action).unsqueeze(-1)
            trigger = F.gumbel_softmax(logits=x[1].logits, tau=0.1, hard=True)
            trigger_entropy = x[1].entropy().unsqueeze(-1)
            log_prob = torch.cat([log_prob, -trigger_entropy], dim=-1)
            action = torch.cat([action, trigger], dim=-1)

        return (action, log_prob)

class RLCriticNet_DriveHard(nn.Module):
    def __init__(self, context_size, particle_size):
        super(RLCriticNet_DriveHard, self).__init__()

        self.context_size = context_size
        self.particle_size = particle_size
        self.num_exo_agents = 10
        self.ego_size = 6 + 4 + 1
        self.exo_size = 6 + 4

        self.exo_fc1_1 = nn.Linear(self.exo_size, 64)
        self.exo_fc2_1 = nn.Linear(64, 64)
        self.exo_fc3_1 = nn.Linear(64, 64)

        self.particle_fc1_1 = nn.Linear(self.ego_size + 64, 128)
        self.particle_fc2_1 = nn.Linear(128, 128)
        self.particle_fc3_1 = nn.Linear(128, 128)

        self.fc3_1 = nn.Linear(context_size + 128 + 2, 256)
        self.fc4_1 = nn.Linear(256, 256)
        self.fc5_1 = nn.Linear(256, 1)

        self.exo_fc1_2 = nn.Linear(self.exo_size, 64)
        self.exo_fc2_2 = nn.Linear(64, 64)
        self.exo_fc3_2 = nn.Linear(64, 64)

        self.particle_fc1_2 = nn.Linear(self.ego_size + 64, 128)
        self.particle_fc2_2 = nn.Linear(128, 128)
        self.particle_fc3_2 = nn.Linear(128, 128)

        self.fc3_2 = nn.Linear(context_size + 128 + 2, 256)
        self.fc4_2 = nn.Linear(256, 256)
        self.fc5_2 = nn.Linear(256, 1)

    def forward(self, x, action):

        (c, x) = x

        x = x.reshape((x.shape[0], -1, self.ego_size + self.num_exo_agents * self.exo_size))
        x_ego = x[...,:self.ego_size]
        x_exo = x[...,self.ego_size:].reshape((x.shape[0], -1, self.num_exo_agents, self.exo_size))

        x_exo_1 = F.relu(self.exo_fc1_1(x_exo))
        x_exo_1 = F.relu(self.exo_fc2_1(x_exo_1))
        x_exo_1 = F.relu(self.exo_fc3_1(x_exo_1))
        x_exo_1 = x_exo_1.mean(dim=-2) # Merge across exo agents.
        x_particle_1 = torch.cat([x_ego, x_exo_1], dim=-1)
        x_particle_1 = F.relu(self.particle_fc1_1(x_particle_1))
        x_particle_1 = F.relu(self.particle_fc2_1(x_particle_1))
        x_particle_1 = F.relu(self.particle_fc3_1(x_particle_1))
        x_particle_1 = x_particle_1.mean(dim=-2) # Merge across particles.
        x1 = torch.cat([c, x_particle_1, action], dim=-1)
        x1 = F.relu(self.fc3_1(x1))
        x1 = F.relu(self.fc4_1(x1))
        x1 = self.fc5_1(x1)

        x_exo_2 = F.relu(self.exo_fc1_2(x_exo))
        x_exo_2 = F.relu(self.exo_fc2_2(x_exo_2))
        x_exo_2 = F.relu(self.exo_fc3_2(x_exo_2))
        x_exo_2 = x_exo_2.mean(dim=-2) # Merge across exo agents.
        x_particle_2 = torch.cat([x_ego, x_exo_2], dim=-1)
        x_particle_2 = F.relu(self.particle_fc1_2(x_particle_2))
        x_particle_2 = F.relu(self.particle_fc2_2(x_particle_2))
        x_particle_2 = F.relu(self.particle_fc3_2(x_particle_2))
        x_particle_2 = x_particle_2.mean(dim=-2) # Merge across particles.
        x2 = torch.cat([c, x_particle_2, action], dim=-1)
        x2 = F.relu(self.fc3_2(x2))
        x2 = F.relu(self.fc4_2(x2))
        x2 = self.fc5_2(x2)

        return (x1, x2)

class RLActorNet_DriveHard(nn.Module):
    def __init__(self, context_size, particle_size):
        super(RLActorNet_DriveHard, self).__init__()

        self.context_size = context_size
        self.particle_size = particle_size
        self.num_exo_agents = 10
        self.ego_size = 6 + 4 + 1
        self.exo_size = 6 + 4

        self.exo_fc1 = nn.Linear(self.exo_size, 64)
        self.exo_fc2 = nn.Linear(64, 64)
        self.exo_fc3 = nn.Linear(64, 64)

        self.particle_fc1 = nn.Linear(self.ego_size + 64, 128)
        self.particle_fc2 = nn.Linear(128, 128)
        self.particle_fc3 = nn.Linear(128, 128)

        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5_action_mean = nn.Linear(256, 2)
        self.fc5_action_std = nn.Linear(256, 2)

    def forward(self, x):

        (c, x) = x

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
        x_particle = x_particle.mean(dim=-2) # Merge across particles.

        x = torch.cat([c, x_particle], dim=-1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x_action_mean = self.fc5_action_mean(x)
        x_action_std = 1e-6 + F.softplus(self.fc5_action_std(x))

        return (x_action_mean, x_action_std)

    def sample(self, x):

        (mean, std) = self.forward(x)
        normal = distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return (action, log_prob)
