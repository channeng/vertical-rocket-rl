import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, help="Name of the run")
parser.add_argument("-l", "--level", default="0", help="Level number")
args = parser.parse_args()

MODEL_NAME = "ppo-" + args.name
all_saves_dir = [
    dir for dir in os.listdir(os.path.join("models", MODEL_NAME))
    if dir.startswith("saves-lvl")]
if len(all_saves_dir) > 0:
    save_path = os.path.join("models", MODEL_NAME, f"saves-lvl{args.level}")
else:
    save_path = os.path.join("models", MODEL_NAME, "saves")
logs_path = os.path.join("models", MODEL_NAME, "logs_eval")

latest_model = sorted([i for i in os.listdir(
    save_path) if i.startswith("best_")])[-1]
latest_model_path = os.path.join(save_path, latest_model)

print(f"Evaluating model: {latest_model_path}")
model = PPO.load(latest_model_path)

# Eval 10 episodes
ENV_ID = f"VerticalRocket-v1-lvl{args.level}"
rewards = []
for i in range(10):
    eval_env = Monitor(gym.make(ENV_ID))
    check_env(eval_env)
    reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=1,
        deterministic=False,
    )
    rewards.append(reward)
    print(reward)
    eval_env.close()

rewards = np.array(rewards)
print(f"mean_reward={rewards.mean():.2f} +/- {rewards.std():.2f}")

# Sample an episode
eval_env = Monitor(gym.make(ENV_ID))
check_env(eval_env)
obs, _ = eval_env.reset()
total_reward = 0.0
total_steps = 0
for i in range(1500):
    eval_env.render()
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, _, info = eval_env.step(action)
    total_reward += reward
    total_steps += 1
    if done:
        obs = eval_env.reset()
        break

print("In %d steps we got %.3f reward" % (total_steps, total_reward))
eval_env.close()
