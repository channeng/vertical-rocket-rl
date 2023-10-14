import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, help="Name of the run")
args = parser.parse_args()
save_path = os.path.join("saves", "ppo-" + args.name)
os.makedirs(save_path, exist_ok=True)

# I haven't tested call back yet.
# from stable_baselines3.common.callbacks import CheckpointCallback

# # Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(
#   save_freq=1000,
#   save_path="./logs/",
#   name_prefix="ppo-" + args.name,
#   save_replay_buffer=True,
#   save_vecnormalize=True,
# )

ENV_ID = "VerticalRocket-v1"
env = gym.make(ENV_ID)
model = PPO(
    "MlpPolicy", env,
    learning_rate=0.0003, n_steps=1000, batch_size=32, n_epochs=50, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1, target_kl=None, stats_window_size=100,
    verbose=1)
    # callback=checkpoint_callback)

eval_env = gym.make(ENV_ID)

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

# Train
model.learn(total_timesteps=100_000)

# Eval
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True,
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
fname = os.path.join(save_path, f"model_{mean_reward:.2f}")
print(f"Model saved to : {fname}")
model.save(fname)

# Replay
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


for i, mean_reward, std_reward in rewards:
    print(f"iteration: {i:00}, mean_reward={mean_reward:.2f} +/- {std_reward}")

# After 10 epochs, rocket takes about 406 steps to 'done', with mean reward of 4.68

# for i in range(10):
#     model.learn(total_timesteps=100_000)

# -----------------------------------------
# | rollout/                |             |
# |    ep_len_mean          | 406         |
# |    ep_rew_mean          | 4.68        |
# | time/                   |             |
# |    fps                  | 1090        |
# |    iterations           | 49          |
# |    time_elapsed         | 92          |
# |    total_timesteps      | 100352      |
# | train/                  |             |
# |    approx_kl            | 0.010291969 |
# |    clip_fraction        | 0.121       |
# |    clip_range           | 0.2         |
# |    entropy_loss         | -3.04       |
# |    explained_variance   | 0.768       |
# |    learning_rate        | 0.0003      |
# |    loss                 | -0.0264     |
# |    n_updates            | 4890        |
# |    policy_gradient_loss | -0.0049     |
# |    std                  | 0.698       |
# |    value_loss           | 0.0507      |
# -----------------------------------------


# Simulation: In 393 steps we got 3.000 reward

# obs = env.reset()
# total_reward = 0.0
# total_steps = 0
# for i in range(1000):
#     env.render()
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     total_reward += reward
#     total_steps += 1
#     if done:
#         obs = env.reset()
#         break
