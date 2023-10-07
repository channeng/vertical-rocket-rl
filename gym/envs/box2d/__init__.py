try:
    import Box2D
    from gym.envs.box2d.rocket_lander import VerticalRocket
except ImportError:
    Box2D = None
