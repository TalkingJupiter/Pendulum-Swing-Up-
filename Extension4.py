import numpy as np
import torch as th
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

from torch.distributions import Normal

from Modules import RecurrentActor
from gae import build_critic


def partial_obs(obs):
    """
    Pendulum full observation:
        [cos(theta), sin(theta), theta_dot]

    Partial observation:
        [cos(theta), sin(theta)]
    """
    return obs[:2].astype(np.float32)


def discounted_returns(rewards, dones, gamma):
    returns = np.zeros((len(rewards), 1), dtype=np.float32)
    running_return = 0.0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            running_return = 0.0

        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def collect_recurrent_rollout(env, policy, steps_per_iter, hidden_size):
    states = []
    hiddens = []
    actions = []
    rewards = []
    dones = []
    old_log_probs = []

    obs, info = env.reset()
    hidden = th.zeros(1, hidden_size)

    episode_return = 0.0
    completed_returns = []

    for _ in range(steps_per_iter):
        obs_partial = partial_obs(obs)

        obs_t = th.as_tensor(obs_partial, dtype=th.float32).unsqueeze(0)

        with th.no_grad():
            mu, sigma, next_hidden = policy(obs_t, hidden)
            dist = Normal(mu, sigma)

            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t).sum(dim=-1, keepdim=True)

        action = action_t.squeeze(0).detach().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        states.append(obs_partial)
        hiddens.append(hidden.squeeze(0).detach().numpy())
        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        old_log_probs.append(log_prob_t.item())

        episode_return += reward

        if done:
            completed_returns.append(episode_return)
            episode_return = 0.0

            obs, info = env.reset()
            hidden = th.zeros(1, hidden_size)
        else:
            obs = next_obs
            hidden = next_hidden.detach()

    # Prevent return calculation from leaking outside this rollout
    dones[-1] = True

    data = {
        "states": np.array(states, dtype=np.float32),
        "hiddens": np.array(hiddens, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
        "old_log_probs": np.array(old_log_probs, dtype=np.float32).reshape(-1, 1),
    }

    if len(completed_returns) > 0:
        avg_return = float(np.mean(completed_returns))
    else:
        avg_return = float(np.sum(rewards))

    return data, avg_return


def train_recurrent_ppo(
    iterations=50,
    steps_per_iter=2048,
    sgd_epochs=10,
    minibatch_size=64,
    learning_rate=3e-4,
    hidden_size=64,
    gamma=0.99,
    eps_clip=0.2,
    c1=0.5,
    c2=0.01,
):
    env = gym.make("Pendulum-v1")

    obs_dim = 2      # partial observation: cos(theta), sin(theta)
    action_dim = env.action_space.shape[0]

    policy = RecurrentActor(obs_dim, action_dim, hidden_size)
    critic = build_critic(obs_dim, hidden_size)

    optimizer = th.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()),
        lr=learning_rate
    )

    returns_per_iter = []
    losses_per_iter = []

    for iteration in range(iterations):
        data, avg_return = collect_recurrent_rollout(
            env,
            policy,
            steps_per_iter,
            hidden_size
        )

        states_t = th.as_tensor(data["states"], dtype=th.float32)
        hiddens_t = th.as_tensor(data["hiddens"], dtype=th.float32)
        actions_t = th.as_tensor(data["actions"], dtype=th.float32)
        old_log_probs_t = th.as_tensor(data["old_log_probs"], dtype=th.float32)

        returns_np = discounted_returns(
            data["rewards"],
            data["dones"],
            gamma
        )

        returns_t = th.as_tensor(returns_np, dtype=th.float32)

        with th.no_grad():
            values_t = critic(states_t)
            advantages_t = returns_t - values_t

            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )

        n = states_t.shape[0]
        losses = []

        for _ in range(sgd_epochs):
            indices = np.random.permutation(n)

            for start in range(0, n, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                mb_states = states_t[mb_idx]
                mb_hiddens = hiddens_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]

                mu, sigma, _ = policy(mb_states, mb_hiddens)
                dist = Normal(mu, sigma)

                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1, keepdim=True)

                ratio = th.exp(new_log_probs - mb_old_log_probs)

                unclipped = ratio * mb_advantages
                clipped = th.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * mb_advantages

                policy_loss = -th.min(unclipped, clipped).mean()

                value_pred = critic(mb_states)
                value_loss = F.mse_loss(value_pred, mb_returns)

                entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()

                loss = policy_loss + c1 * value_loss - c2 * entropy

                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(critic.parameters()),
                    max_norm=0.5
                )
                optimizer.step()

                losses.append(loss.item())

        returns_per_iter.append(avg_return)
        losses_per_iter.append(float(np.mean(losses)))

        print(
            f"iter {iteration + 1}/{iterations} | "
            f"partial obs + GRU | "
            f"return={avg_return:.2f} | "
            f"loss={losses_per_iter[-1]:.4f}"
        )

    env.close()

    return policy, returns_per_iter, losses_per_iter


if __name__ == "__main__":
    policy, returns, losses = train_recurrent_ppo(
        iterations=50,
        steps_per_iter=2048
    )

    plt.figure()
    plt.plot(returns, label="Partial observation + GRU policy")
    plt.xlabel("Training iteration")
    plt.ylabel("Episodic return")
    plt.title("Extension 4: Partial Observability with Recurrent Policy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(losses, label="Recurrent PPO loss")
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.title("Extension 4: Recurrent PPO Loss")
    plt.legend()
    plt.grid(True)
    plt.show()