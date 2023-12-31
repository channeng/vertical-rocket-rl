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
parser.add_argument("-m", "--model", default="", help="Model")
args = parser.parse_args()

MODEL_NAME = "ppo-" + args.name

save_path = os.path.join("models", MODEL_NAME, "saves" + args.model)
latest_model = sorted([i for i in os.listdir(
    save_path) if i.startswith("best_")])[-1]
latest_model_path = os.path.join(save_path, latest_model)

print(f"Evaluating model: {latest_model_path}")
model = PPO.load(latest_model_path, device="cpu")

ENV_ID = f"VerticalRocket-v1-lvl{args.level}"
eval_env = Monitor(gym.make(ENV_ID))
check_env(eval_env)
obs, _ = eval_env.reset()
total_reward = 0.0
total_steps = 0
for i in range(2000):
    eval_env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = eval_env.step(action)
    total_reward += reward
    total_steps += 1
    if done:
        print(info)
        print(total_reward)
        obs = eval_env.reset()
        break

print("In %d steps we got %.3f reward" % (total_steps, total_reward))
eval_env.close()
