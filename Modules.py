"""Utility modules provided for the assignment.

`NormalModule` is the final head of the policy network: it takes a feature
vector and returns (mu, sigma) parameterising a Gaussian action distribution.
The mean is tanh-squashed into (-1, 1); you will need to rescale it into the
env's action range before stepping the environment. The log standard
deviation is a learnable but state-independent parameter.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalModule(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.m = nn.Linear(inp, out)
        log_std = -0.5 * np.ones(out, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, inputs):
        mout = self.m(inputs)
        vout = torch.exp(self.log_std)
        # mu is squashed to (-1, 1); rescale it to the env action range later.
        return F.tanh(mout), vout
    

class StateDependentNormalModule(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.m = nn.Linear(inp, out)
        self.std_layer = nn.Linear(inp, out)

        nn.init.constant_(self.std_layer.weight, 0.0)
        nn.init.constant_(self.std_layer.bias, -0.5)
    
    def forward(self, inputs):
        mu = torch.tanh(self.m(inputs))
        log_std = self.std_layer(inputs)
        log_std = torch.clamp(log_std, -5.0, 2.0)

        sigma = torch.exp(log_std)

        return mu, sigma

class RecurrentActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(obs_dim, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5) # Fixed log std for this extension

    def forward(self, obs, hidden=None):
        single_input = False

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single_input = True

        batch_size = obs.shape[0]

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=obs.device)

        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)

        next_hidden = self.gru(obs, hidden)

        mu = torch.tanh(self.mean_layer(next_hidden))
        sigma = torch.exp(self.log_std).expand_as(mu)

        if single_input:
            mu = mu.squeeze(0)
            sigma = sigma.squeeze(0)
            next_hidden = next_hidden.squeeze(0)

        return mu, sigma, next_hidden