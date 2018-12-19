import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    h_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(h_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor model (=Policy)"""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_params()

    def reset_params(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh((self.fc3(x)))


class Critic(nn.Module):
    """ Critic Model (=Value)"""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fcs2_units=128):
        """Initialize parameters and build model"""
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fcs2 = nn.Linear(fcs1_units+action_size, fcs2_units)
        self.fcs3 = nn.Linear(fcs2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fcs2.weight.data.uniform_(*hidden_init(self.fcs2))
        self.fcs3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic network"""
        xs = F.relu((self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fcs2(x))
        return self.fcs3(x)
