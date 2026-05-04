from ppo import train_ppo
from plotting import plot_learning_curves, plot_loss_curves

policy_base, ret_base, loss_base = train_ppo(iterations=50, state_dep_std=False)
policy_std, ret_std, loss_std = train_ppo(iterations=50, state_dep_std=True)

plot_learning_curves({"Base PPO(fixed std)": ret_base, "PPO state dep std": ret_std}, title="Extenstion 1: State Dependent Standart Deviation")
plot_loss_curves({"Base PPO(fixed std)": loss_base, "PPO state dep std": loss_std}, title="Extenstion 1: PPO loss comperison")

