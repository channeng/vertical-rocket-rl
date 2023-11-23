import os
import argparse
from typing import Callable

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, help="Name of the run")
args = parser.parse_args()

MODEL_NAME = "ppo-" + args.name

checkpoints_path = os.path.join("models", MODEL_NAME, "checkpoints")
save_path = os.path.join("models", MODEL_NAME, "saves")
logs_path = os.path.join("models", MODEL_NAME, "logs")
eval_path = os.path.join("models", MODEL_NAME, "eval")
os.makedirs(save_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)

LR_INIT = 3e-4
LR_FINAL = 3e-4


def linear_schedule(initial_value: float = LR_INIT) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > 0.2:
            return LR_INIT
        return progress_remaining / 0.2 * (initial_value - LR_FINAL) + LR_FINAL

    return func


level = 0
eval_level = 6

env = make_vec_env(f"VerticalRocket-v1-lvl{level}", n_envs=4)
eval_env = Monitor(gym.make(f"VerticalRocket-v1-lvl{eval_level}"))

model = PPO(
    "MlpPolicy", env,
    learning_rate=linear_schedule(LR_INIT),
    n_steps=2048, batch_size=128, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None,
    normalize_advantage=True, ent_coef=0, vf_coef=0.5, max_grad_norm=0.5,
    use_sde=False, sde_sample_freq=-1, target_kl=None, stats_window_size=100,
    tensorboard_log=logs_path, device="cpu",
    verbose=1
)

N_EVAL_EPISODES = 50
EVAL_FREQUENCY = 10_000
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save_path + f"-lvl{eval_level}",
    log_path=eval_path,
    eval_freq=EVAL_FREQUENCY,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False
)

highest_reward = -np.inf
plateau = 0

log_file = open(os.path.join(logs_path, "log_curriculum-learning.csv"), 'w')
log_file.write("iteration,mean_reward,level\n")

while True:
    print(f">>>>>>>>>>>>>>>>>> Training with the curriculum level {level}...")
    model.learn(
        total_timesteps=50_000,
        reset_num_timesteps=False,
        progress_bar=True,
        callback=[
            eval_callback,
        ])

    eval_file = os.path.join(eval_path, "evaluations.npz")
    data = np.load(eval_file)

    timestamp = data['timesteps'][-1]
    mean_reward = data['results'].mean(axis=1).max()

    log_file.write(
        f"{timestamp},{mean_reward},{level}\n")
    log_file.flush()

    if mean_reward > highest_reward:
        plateau = 0
        highest_reward = mean_reward
    else:
        plateau += 1
        print(f'>>>>>>>>>>>>>>>>>> Plateau {plateau}...')

        plateau_cap = 5
        if level == eval_level:
            plateau_cap = 20
        if plateau >= plateau_cap:
            plateau = 0
            level += 1
            print(f'>>>>>>>>>>>>>>>>>> Going to the next level {level}...')
            if level > eval_level:
                break

            env = make_vec_env(f"VerticalRocket-v1-lvl{level}", n_envs=4)
            env.reset()
            model.env = env

# def train(model, eval_env, level_number):
#     # Callbacks
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=save_path + f"-lvl{level_number}",
#         log_path=eval_path,
#         eval_freq=EVAL_FREQUENCY,
#         n_eval_episodes=N_EVAL_EPISODES,
#         deterministic=False,
#         render=False
#     )

#     # Train
#     model.learn(
#         total_timesteps=LEVEL_TRAIN_TIMESTEPS[level_number],
#         callback=[
#             eval_callback,
#         ])

#     # Print eval stats
#     eval_file = os.path.join(eval_path, "evaluations.npz")
#     rename_eval_filename = f"evaluations_lvl{level_number}.npz"
#     rename_eval_file = os.path.join(eval_path, rename_eval_filename)
#     os.rename(eval_file, rename_eval_file)
#     data = np.load(rename_eval_file)

#     success_rate = data["successes"].sum(axis=1) / data["successes"].shape[1]
#     mean_reward = data["results"].mean(axis=1)
#     mean_ep_lengths = data["ep_lengths"].mean(axis=1)
#     for i in range(len(data["timesteps"])):
#         print(
#             f"{data['timesteps'][i]:>7} -> "
#             f"success rate : {success_rate[i]:>3.0%}, "
#             f"mean reward  : {mean_reward[i]:>5.2f}, "
#             f"mean ep length: {mean_ep_lengths[i]:>5.0f}"
#         )


# initial_level = LEVELS[1]
# ENV_ID = f"VerticalRocket-v1-lvl{initial_level}"
# print(f"Training with {ENV_ID}\n\n")

# env = make_vec_env(ENV_ID, n_envs=4)
# eval_env = gym.make(ENV_ID)
# eval_env = Monitor(eval_env)
# model = PPO(
#     "MlpPolicy", env,
#     # learning_rate=LR_INIT,
#     learning_rate=linear_schedule(LR_INIT),
#     n_steps=2048, batch_size=128, n_epochs=10,
#     gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None,
#     normalize_advantage=True, ent_coef=0, vf_coef=0.5, max_grad_norm=0.5,
#     use_sde=False, sde_sample_freq=-1, target_kl=None, stats_window_size=100,
#     tensorboard_log=logs_path, device="cpu",
#     verbose=1
# )
# train(model, eval_env, initial_level)
# env.close()
# eval_env.close()

# for level_number in LEVELS[1:]:
#     ENV_ID = f"VerticalRocket-v1-lvl{level_number}"
#     print(f"Training with {ENV_ID}\n\n")

#     env = make_vec_env(ENV_ID, n_envs=4)
#     eval_env = gym.make(ENV_ID)
#     eval_env = Monitor(eval_env)

#     model.set_env(env)
#     train(model, eval_env, level_number)

#     env.close()
#     eval_env.close()
