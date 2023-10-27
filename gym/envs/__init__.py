from gymnasium.envs.registration import register

register(
    id='VerticalRocket-v1',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10
)

register(
    id='VerticalRocket-v1-lvl1',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 1},
)

register(
    id='VerticalRocket-v1-lvl2',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 2},
)

print(">>>>>>>>>>>>>>>>>>>> Registered VerticalRocket-v1 <<<<<<<<<<<<<<<<<<<<")
