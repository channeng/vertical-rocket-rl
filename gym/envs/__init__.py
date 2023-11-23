from gymnasium.envs.registration import register

register(
    id='VerticalRocket-v1-lvl0',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 0},
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

register(
    id='VerticalRocket-v1-lvl3',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 3},
)

register(
    id='VerticalRocket-v1-lvl4',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 4},
)

register(
    id='VerticalRocket-v1-lvl5',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 5},
)

register(
    id='VerticalRocket-v1-lvl6',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 6},
)

register(
    id='VerticalRocket-v1-lvl7',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 7},
)

register(
    id='VerticalRocket-v1-lvl8',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 8},
)
register(
    id='VerticalRocket-v1-lvl9',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'level_number': 9},
)

print(">>>>>>>>>>>>>>>>>>>> Registered VerticalRocket-v1 <<<<<<<<<<<<<<<<<<<<")
