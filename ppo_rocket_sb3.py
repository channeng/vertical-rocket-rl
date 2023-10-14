import gym
from gym import wrappers
from stable_baselines3 import PPO

ENV_ID = "VerticalRocket-v1"
env = gym.make(ENV_ID)
model = PPO("MlpPolicy", env, verbose=1)

for i in range(10):
    model.learn(total_timesteps=100_000)

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
