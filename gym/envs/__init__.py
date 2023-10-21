from gym.envs.registration import register, make, spec

register(
    id='VerticalRocket-v1',
    entry_point='gym.envs.box2d:VerticalRocket',
    max_episode_steps=1000,
    reward_threshold=8,
)

print(">>>>>>>>>>>>>>>>>>>> Registered VerticalRocket-v1 <<<<<<<<<<<<<<<<<<<<")
