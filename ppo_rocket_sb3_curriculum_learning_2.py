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
test_level = 1

train_env = make_vec_env(f"VerticalRocket-v1-lvl{level}", n_envs=4)
model = PPO(
    "MlpPolicy", train_env,
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
eval_env = Monitor(gym.make(f"VerticalRocket-v1-lvl{level}"))
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save_path + f"-lvl{level}",
    log_path=eval_path,
    eval_freq=EVAL_FREQUENCY,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False
)


def play(env, model, n_steps=2000):
    obs, _ = env.reset()
    total_reward = 0
    episode_length = 0
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        total_reward += reward
        episode_length += 1

        if done:
            break

    return total_reward, episode_length, info['is_success']


agg_test_logger = open(os.path.join(
    logs_path, "log_agg_curriculum-learning.csv"), 'w')
agg_test_logger.write(
    "iteration,level,test_mean_reward,test_mean_episode_length,test_success_rate\n")

test_logger = open(os.path.join(
    logs_path, "log_curriculum-learning.csv"), 'w')
test_logger.write(
    "iteration,level,test_reward,test_episode_length,test_successful\n")

test_env = Monitor(gym.make(f"VerticalRocket-v1-lvl{test_level}"))
highest_reward = -np.inf
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
    evaluations = np.load(eval_file)
    timestamp = evaluations['timesteps'][-1]

    rewards = []
    episode_lengths = []
    successes = []
    for _ in range(20):
        total_reward, episode_length, successful = play(test_env, model)
        rewards.append(total_reward)
        episode_lengths.append(episode_length)
        successes.append(successful)

        test_logger.write(
            f"{timestamp},{level},{total_reward},{episode_length},{successful}\n")
    test_logger.flush()

    mean_reward = np.mean(rewards)
    mean_episode_length = np.mean(episode_lengths)
    success_rate = np.mean(successes)

    agg_test_logger.write(
        f"{timestamp},{level},{mean_reward},{mean_episode_length},{success_rate}\n")
    agg_test_logger.flush()

    if mean_reward > highest_reward:
        highest_reward = mean_reward
        print(f">>>>> Found a better model:")
        print(f">>>>> Timestamp: {timestamp}")
        print(f">>>>> Level: {level}")
        print(f">>>>> Mean reward: {mean_reward}")
        print(f">>>>> Mean episode length: {mean_episode_length}")
        print(f">>>>> Success rate: {success_rate}")
        model.save(os.path.join(save_path, f"best_curriculum_learning_model"))

    eval_mean_rewards = evaluations['results'].mean(axis=1)
    eval_max_mean_reward = eval_mean_rewards.max()
    success_rate = evaluations['successes'].sum(
        axis=1) / evaluations['successes'].shape[1]
    if np.any(eval_mean_rewards[-20:] == eval_max_mean_reward) == False or np.any(success_rate == 1.0):
        os.rename(os.path.join(eval_path, f"evaluations.npz"),
                  os.path.join(eval_path, f"evaluations_level{level}.npz"))
        level += 1
        if level > test_level:
            break
        print(f'>>>>>>>>>>>>>>>>>> Going to the next level {level}...')

        train_env = make_vec_env(f"VerticalRocket-v1-lvl{level}", n_envs=4)
        train_env.reset()
        model.env = train_env

        eval_env = Monitor(gym.make(f"VerticalRocket-v1-lvl{level}"))
        eval_env.reset()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path + f"-lvl{level}",
            log_path=eval_path,
            eval_freq=EVAL_FREQUENCY,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False
        )
