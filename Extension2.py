from ppo import train_ppo
from plotting import plot_learning_curves, plot_loss_curves

policy_base, ret_base, loss_base = train_ppo(iterations=50, ensamble_crit=False)
policy_ens, ret_ens, loss_ens = train_ppo(iterations=50, ensamble_crit=True, num_crit=3)

plot_learning_curves({"Base PPO single critic": ret_base, "PPO ensemble critics": ret_ens}, title="Extension 2: Ensemble Critics Return Comparison")
plot_loss_curves({"Base PPO single critic": loss_base, "PPO ensemble critics": loss_ens }, title="Extension 2: PPO Loss Comparison")