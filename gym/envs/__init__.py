from gymnasium.envs.registration import register


register(
    id='VerticalRocket-v1-lvl0',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 0,
            'has_additional_constraints': True, 'starting_height': 400},
)

register(
    id='VerticalRocket-v1-lvl1',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 1,
            'has_additional_constraints': True, 'starting_height': 400},
)

register(
    id='VerticalRocket-v1-lvl2',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 2,
            'has_additional_constraints': True, 'starting_height': 400},
)

register(
    id='VerticalRocket-v1-lvl3',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 3,
            'has_additional_constraints': True, 'starting_height': 400},
)

register(
    id='VerticalRocket-v1-lvl4',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 4,
            'has_additional_constraints': True, 'starting_height': 400},
)

register(
    id='VerticalRocket-v1-lvl5',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 5,
            'has_additional_constraints': True, 'starting_height': 400},
)

register(
    id='VerticalRocket-v1-lvl6',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 6,
            'has_additional_constraints': True, 'starting_height': 400},
)

register(
    id='VerticalRocket-v1-lvl7',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 7,
            'has_additional_constraints': True, 'starting_height': 400},
)

register(
    id='VerticalRocket-v1-lvl8',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 8,
            'has_additional_constraints': True, 'starting_height': 400},
)
register(
    id='VerticalRocket-v1-lvl9',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': True, 'level_number': 9,
            'has_additional_constraints': True, 'starting_height': 400},
)

register(
    id='VerticalRocket-v1-stage1',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 1, 'curriculum': False, 'level_number': 0,
            'has_additional_constraints': False, 'starting_height': 300},
)

register(
    id='VerticalRocket-v1-stage2',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 2, 'curriculum': False, 'level_number': 0,
            'has_additional_constraints': False, 'starting_height': 300},
)

register(
    id='VerticalRocket-v1-stage3',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 3, 'curriculum': False, 'level_number': 0,
            'has_additional_constraints': False, 'starting_height': 300},
)

register(
    id='VerticalRocket-v1-stage4',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=2000,
    reward_threshold=10,
    kwargs={'stage': 4, 'curriculum': False, 'level_number': 0,
            'has_additional_constraints': False, 'starting_height': 300},
)
print(">>>>>>>>>>>>>>>>>>>> Registered VerticalRocket-v1 <<<<<<<<<<<<<<<<<<<<")
