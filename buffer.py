"""
HW4 — Task 1: Replay buffer and environment interaction.

Complete the four TODO items below before moving on to vpg.py.
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
from gymnasium.spaces import Box

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
        self.next_states = np.zeros((size, sdim), dtype=sdtype)
        self.i     = 0
        self.size  = size
        self.max_i = 0
        self.ep_len = ep_len

    def add(self, state, action, reward, next_states, done):
        if self.i >= self.size: return #Check if we are oversize
        current_row = self.i 

        self.states[current_row] = state
        self.actions[current_row] = action
        self.rewards[current_row] = reward
        self.next_states[current_row] = next_states
        self.dones[current_row] = done

        self.i += 1 #Update the location
        self.max_i += 1 
        

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.max_i, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
            self.ret_to_go[idxs],
        )

    def calc_reward_to_go(self, gamma):
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
        raw_action = act(agent, observation)
        env_action = np.clip(rescale_actions(raw_action, env.action_space.low, env.action_space.high), env.action_space.low, env.action_space.high)
        next_observation, reward, terminated, truncated, info = env.step(env_action)
        if truncated or terminated:
            done = True

        buffer.add(current_state, raw_action, reward, next_observation, done)

        if done:
            observation, info = env.reset()
            done = False
        else:
            observation = next_observation

    
    # buffer.calc_reward_to_go()
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

# -------------------
# PROJECT EXTENSION 3
# -------------------
def act_batch(policy, state):
    state_t = th.as_tensor(state, dtype=th.float32)

    mu, sigma = policy(state_t)
    dist = Normal(mu, sigma)

    actions = dist.sample()
    actions = actions.detach().numpy()
    return actions


def collect_parallel(size, agent, num_envs=4, title="collecting parallel"):
    envs = gym.vector.SyncVectorEnv([
        lambda: gym.make("Pendulum-v1")
        for _ in range(num_envs)
        ])
    
    sdim = envs.single_observation_space.shape[0]
    adim = envs.single_action_space.shape[0]

    action_space = envs.single_action_space
    assert isinstance(action_space, Box)

    num_steps = int(np.ceil(size/num_envs))

    traj =[]
    for i in range(num_envs):
        traj.append({
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        })
    
    observ, infos = envs.reset()
    for i in range(num_steps):
        current_states = observ.copy()
        raw_actions = act_batch(agent, current_states)
        env_actions = np.clip(
            rescale_actions(raw_actions, action_space.low, action_space.high),
            action_space.low,
            action_space.high,
        )
        next_observ, rewards, terminated, truncated, infos = envs.step(env_actions)

        dones = np.logical_or(terminated, truncated)

        for idx in range(num_envs):
            traj[idx]["states"].append(current_states[idx].copy())
            traj[idx]["actions"].append(raw_actions[idx].copy())
            traj[idx]["rewards"].append(rewards[idx].copy())
            traj[idx]["next_states"].append(next_observ[idx].copy())
            traj[idx]["dones"].append(dones[idx].copy())

        observ = next_observ
    envs.close()

    buffer = Buffer(sdim, adim, num_steps * num_envs)

    for idx in range(num_envs):
        traj[idx]["dones"][-1] = True #prevents adv calc from leaking into next env traj

        for t in range(num_steps):
            buffer.add(
                traj[idx]["states"][t],
                traj[idx]["actions"][t],
                traj[idx]["rewards"][t],
                traj[idx]["next_states"][t],
                traj[idx]["dones"][t],            
            )
    
    awg_rwd = np.mean(buffer.rewards[:buffer.max_i])

    return buffer, awg_rwd
