"""
HW4 — Task 1: Replay buffer and environment interaction.

Complete the four TODO items below before moving on to vpg.py.
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym

from Modules import NormalModule


class Buffer:
    """Experience replay buffer storing one-step transitions.

    Use-contract:
        add(state, action, reward, done)             — push one transition
        calc_reward_to_go(gamma)                     — fill self.ret_to_go
        sample(batch_size) -> tuple of numpy arrays  — draw a mini-batch
    """

    def __init__(self, sdim, adim, size, sdtype=np.float32, adtype=np.float32, ep_len=200):
        self.states    = np.zeros((size, sdim), dtype=sdtype)
        self.actions   = np.zeros((size, adim), dtype=adtype)
        self.rewards   = np.zeros((size, 1),    dtype=np.float32)
        self.ret_to_go = np.zeros((size, 1),    dtype=np.float32)
        self.dones     = np.zeros((size, 1),    dtype=bool)
        self.i     = 0
        self.size  = size
        self.max_i = 0
        self.ep_len = ep_len

    def add(self, state, action, reward, done):
        if self.i >= self.size: return #Check if we are oversize
        current_row = self.i 

        self.states[current_row] = state
        self.actions[current_row] = action
        self.rewards[current_row] = reward
        self.dones[current_row] = done

        self.i += 1 #Update the location
        self.max_i += 1 
        

    def sample(self, batch_size):
        upper = max(self.max_i - 1, 1)
        idxs = np.random.randint(0, upper, size=batch_size)
        done_mask = self.dones[idxs, 0]
        idxs = np.where(done_mask, np.maximum(idxs - 1, 0), idxs)
        next_idxs = idxs + 1
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.states[next_idxs],
            self.dones[next_idxs],
            self.ret_to_go[idxs],
            self.ret_to_go[next_idxs],
        )

    def calc_reward_to_go(self, gamma=0.975):
        reward_return = 0
        for row in range(self.max_i-1, -1, -1):
            if self.dones[row] == True:
                reward_return = 0
        
            reward_return = self.rewards[row] + gamma * reward_return
            self.ret_to_go[row] = reward_return


def collect_data(size, env, agent, title="collecting"):
    """Roll out `agent` (a policy network) in `env` for `size` steps.

    Returns:
        buffer  — a populated Buffer
        avg_rwd — average per-step reward observed during the rollout
    """
    buffer = Buffer(env.observation_space.shape[0], env.action_space.shape[0], size)

    observation, info = env.reset()
    done = False

    for _ in range(size):
        current_state = observation
        action = act(agent, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if truncated or terminated:
            done = True

        buffer.add(current_state, action, reward, done)

        if done:
            observation, info = env.reset()
            done = False
        else:
            observation = next_observation

    
    buffer.calc_reward_to_go()
    return buffer, np.mean(buffer.rewards)


def act(policy, state):
    """Sample a continuous action a ~ N(mu(state), sigma) from the policy."""
    state = th.from_numpy(state).float() #convert np.array to pytorch tensors
    mu, sigma = policy(state) #Receive mu, sigma values from policy

    distribution = Normal(mu, sigma) #Gather the normalized distribution
    
    action = distribution.sample() #Sample from the distribution object
    action = action.detach().numpy() #Stoping the tensor track to convert tensors to numpy array
    
    return action
 


def rescale_actions(action, amin, amax):
    """Rescale a tanh-squashed action from (-1, 1) to the env range [amin, amax]."""
    return amin + 0.5 * (action + 1.0) * (amax - amin)
