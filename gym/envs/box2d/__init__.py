try:
    import Box2D
    from gym.envs.box2d.vertical_rocket import VerticalRocket
except ImportError:
    Box2D = None
