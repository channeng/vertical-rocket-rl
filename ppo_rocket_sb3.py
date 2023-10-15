import os
import numpy as np
import gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, help="Name of the run")
parser.add_argument("-m", "--method", default="ppo", help="ppo or a2c")
args = parser.parse_args()

MODEL_NAME = args.method + "-" + args.name

checkpoints_path = os.path.join("models", MODEL_NAME, "checkpoints")
save_path = os.path.join("models", MODEL_NAME, "saves")
logs_path = os.path.join("models", MODEL_NAME, "logs")
eval_path = os.path.join("models", MODEL_NAME, "eval")
os.makedirs(save_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)

ENV_ID = "VerticalRocket-v1"
env = gym.make(ENV_ID)
eval_env = gym.make(ENV_ID)

if args.method == "ppo":
    model = PPO(
        "MlpPolicy", env,
        learning_rate=1e-5, n_steps=1000, batch_size=250, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None,
        normalize_advantage=True, ent_coef=0, vf_coef=0.5, max_grad_norm=0.5,
        use_sde=False, sde_sample_freq=-1, target_kl=None, stats_window_size=100,
        tensorboard_log=logs_path,
        verbose=1
    )
else:
    model = A2C(
        "MlpPolicy", env,
        tensorboard_log=logs_path,
        verbose=1)

# Initial eval
rewards = []
max_reward = 0
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True,
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
rewards.append((0, mean_reward, std_reward))


# Callbacks
frequency = 20_000
checkpoint_callback = CheckpointCallback(
  save_freq=frequency,
  save_path=checkpoints_path,
  name_prefix="ppo-" + args.name,
  save_replay_buffer=True,
  save_vecnormalize=True,
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save_path,
    log_path=eval_path,
    eval_freq=frequency,
    n_eval_episodes=10,
    deterministic=True,
    render=True
)

# Train
train_total_timesteps = 1_000_000
model.learn(
    total_timesteps=train_total_timesteps,
    callback=[checkpoint_callback, eval_callback])

# Print eval stats
data = np.load(os.path.join(eval_path, "evaluations.npz"))
mean_reward = data["results"].mean(axis=1)
mean_ep_lengths = data["ep_lengths"].mean(axis=1)
for i in range(len(data["timesteps"])):
    print(
        f"{data['timesteps'][i]:>7} -> "
        f"mean reward: {mean_reward[i]:.2f}, "
        f"mean ep length: {mean_ep_lengths[i]:.0f}"
    )
