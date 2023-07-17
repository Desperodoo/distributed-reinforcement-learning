import argparse

ppo_args = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
# Learner and Worker
ppo_args.add_argument("--max_train_steps", type=int, default=int(5e7), help=" Maximum number of training steps")
ppo_args.add_argument("--sample_epi_num", type=int, default=1, help="Sample episode number")
ppo_args.add_argument("--action_max", type=list, default=[0.5, 0.5, 0.5, 0.5], help="Sample episode number")

# Evaluator
ppo_args.add_argument("--evaluate_times", type=float, default=20, help="Evaluate the policy every 'evaluate_freq' steps")
ppo_args.add_argument("--save_freq", type=int, default=20, help="Save frequency")
ppo_args.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
ppo_args.add_argument("--save_cwd", type=str, default=f"./model")

# Runner
ppo_args.add_argument("--learner_device", type=str, default="cuda")
ppo_args.add_argument("--worker_device", type=str, default="cpu")
ppo_args.add_argument("--evaluator_device", type=str, default="cpu")

# Networks
ppo_args.add_argument("--num_layers", type=int, default=2, help="The number of the hidden layers of RNN")
ppo_args.add_argument("--rnn_hidden_dim", type=int, default=256, help="The dimension of the hidden layer of RNN")
ppo_args.add_argument("--mlp_hidden_dim", type=int, default=256, help="The dimension of the hidden layer of MLP")
ppo_args.add_argument("--mlp_output_dim", type=int, default=256, help="The dimension of the hidden layer of MLP")

ppo_args.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
ppo_args.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
ppo_args.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
ppo_args.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
ppo_args.add_argument("--K_epochs", type=int, default=5, help="PPO parameter")

# Tricks
ppo_args.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
ppo_args.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
ppo_args.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
ppo_args.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
ppo_args.add_argument("--entropy_coef", type=float, default=0.05, help="Trick 5: policy entropy")
ppo_args.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
ppo_args.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
ppo_args.add_argument("--use_value_clip", type=float, default=True, help="Whether to use value clip.")
ppo_args.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
ppo_args.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
ppo_args.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
ppo_args.add_argument("--use_sn", type=float, default=True, help="Trick 11: spectral normalization")

ppo_args = ppo_args.parse_args()
