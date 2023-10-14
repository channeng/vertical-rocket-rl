import os
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, help="Name of the run")
args = parser.parse_args()
save_path = os.path.join("saves", "ppo-" + args.name)

latest_model  =sorted([i for i in os.listdir(save_path) if i.startswith("model_")])[-1]
latest_model_path = os.path.join(save_path, latest_model)

print(f"Evaluating model: {latest_model_path}")
model = PPO.load(latest_model_path)

ENV_ID = "VerticalRocket-v1"
env = gym.make(ENV_ID)

mean_reward, std_reward = evaluate_policy(
    model,
	env,
    n_eval_episodes=10,
    deterministic=True,
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

obs = env.reset()
total_reward = 0.0
total_steps = 0
for i in range(1000):
    env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    total_steps += 1
    if done:
        obs = env.reset()
        break

print("In %d steps we got %.3f reward" % (total_steps, total_reward))
env.close()
