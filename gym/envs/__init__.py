from gym.envs.registration import register, make, spec
import gym.envs.box2d.vertical_rocket

register(
    id='VerticalRocket-v1',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=1000,
    reward_threshold=8,
)

print(">>>>>>>>>>>>>>>>>>>> Registered VerticalRocket-v1 <<<<<<<<<<<<<<<<<<<<")
