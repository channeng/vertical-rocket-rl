from gymnasium.envs.registration import register

register(
    id='VerticalRocket-v1',
    entry_point='gym.envs.box2d.vertical_rocket:VerticalRocket',
    max_episode_steps=1500,
    reward_threshold=8
)

print(">>>>>>>>>>>>>>>>>>>> Registered VerticalRocket-v1 <<<<<<<<<<<<<<<<<<<<")
