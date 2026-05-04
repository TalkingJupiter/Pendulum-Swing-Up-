import time
import matplotlib.pyplot as plt

from ppo import train_ppo
from plotting import plot_learning_curves, plot_loss_curves


env_settings = [1, 2, 4, 8]

returns_dict = {}
loss_dict = {}
total_times = {}

for n in env_settings:
    print(f"\nRunning PPO with {n} parallel environment(s)...")

    start = time.perf_counter()

    policy, returns, losses = train_ppo(iterations=50, steps_per_iter=2048, num_envs=n)

    end = time.perf_counter()

    returns_dict[f"{n} envs"] = returns
    loss_dict[f"{n} envs"] = losses
    total_times[n] = end - start

print("\nTotal training times:")
for n in env_settings:
    print(f"{n} envs: {total_times[n]:.2f} seconds")


plot_learning_curves(returns_dict, title="Extension 3: Parallel Data Collection Return Comparison")

plot_loss_curves(loss_dict, title="Extension 3: PPO Loss Comparison")


plt.figure()
plt.plot(env_settings, [total_times[n] for n in env_settings], marker="o")
plt.xlabel("Number of parallel environments")
plt.ylabel("Total training time seconds")
plt.title("Extension 3: Training Time vs Number of Environments")
plt.grid(True)
plt.show()