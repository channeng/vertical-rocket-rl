import numpy as np
import Box2D
from Box2D.b2 import (
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    distanceJointDef,
    contactListener,
)

import gymnasium as gym
from gymnasium import spaces

"""

The objective of this environment is to land a rocket on a ship.

STATE VARIABLES
The state consists of the following variables:
    - x position
    - y position
    - angle
    - first leg ground contact indicator
    - second leg ground contact indicator
    - throttle
    - engine gimbal
If VEL_STATE is set to true, the velocities are included:
    - x velocity
    - y velocity
    - angular velocity
all state variables are roughly in the range [-1, 1]

CONTROL INPUTS
Discrete control inputs are:
    - gimbal left
    - gimbal right
    - throttle up
    - throttle down
    - use first control thruster
    - use second control thruster
    - no action
"""

CONTINUOUS = True
VEL_STATE = True        # Add velocity info to state
FPS = 60
SCALE_S = 0.35          # Temporal Scaling, lower is faster - adjust forces appropriately
# INITIAL_RANDOM = 0.0    # Random scaling of initial velocity, higher is more difficult

# ROCKET
MIN_THROTTLE = 0.4
GIMBAL_THRESHOLD = 0.4
MAIN_ENGINE_POWER = 1600 * SCALE_S * 1.0
SIDE_ENGINE_POWER = 100 / FPS * SCALE_S * 2.0

ROCKET_WIDTH = 3.66 * SCALE_S
ROCKET_HEIGHT = ROCKET_WIDTH / 3.7 * 47.9
ENGINE_HEIGHT = ROCKET_WIDTH * 0.5
ENGINE_WIDTH = ENGINE_HEIGHT * 0.7
THRUSTER_HEIGHT = ROCKET_HEIGHT * 0.86

# LEGS
LEG_LENGTH = ROCKET_WIDTH * 4.2
BASE_ANGLE = -0.27
SPRING_ANGLE = 0.27
LEG_AWAY = ROCKET_WIDTH / 2

# SHIP
SHIP_HEIGHT = ROCKET_WIDTH
SHIP_WIDTH = SHIP_HEIGHT * 80

# VIEWPORT
VIEWPORT_H = 720
VIEWPORT_W = 500


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.water in [contact.fixtureA.body, contact.fixtureB.body]
            or self.env.lander in [contact.fixtureA.body, contact.fixtureB.body]
            or self.env.containers[0] in [contact.fixtureA.body, contact.fixtureB.body]
            or self.env.containers[1] in [contact.fixtureA.body, contact.fixtureB.body]
        ):
            self.env.game_over = True
        else:
            for i in range(2):
                if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class VerticalRocket(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, level_number=0, continuous=True, speed_threshold=1):
        super(VerticalRocket, self).__init__()
        self.level_number = level_number

        if self.level_number >= 2:
            self.START_HEIGHT = 1000.0 * (1 + np.random.uniform(-0.1, 0.1))
            self.START_SPEED = 80.0 * (1 + np.random.uniform(-0.1, 0.1))
        elif self.level_number == 1:
            self.START_HEIGHT = 700.0 * (1 + np.random.uniform(-0.2, 0.2))
            self.START_SPEED = 60.0 * (1 + np.random.uniform(-0.2, 0.2))
        elif self.level_number == 0:
            self.START_HEIGHT = 400.0 * (1 + np.random.uniform(-0.25, 0.25))
            self.START_SPEED = 20.0 * (1 + np.random.uniform(-0.25, 0.25))

        self.H = 1.1 * self.START_HEIGHT * SCALE_S
        self.W = float(VIEWPORT_W) / VIEWPORT_H * self.H

        self.viewer = None
        self.episode_number = 0

        self.world = Box2D.b2World()
        self.water = None
        self.lander = None
        self.engine = None
        self.ship = None
        self.legs = []
        self.state = []
        self.continuous = continuous
        self.landed = False
        self.landed_fraction = []
        self.good_landings = 0
        self.total_landed_ticks = 0
        self.landed_ticks = 0
        self.done = False
        self.speed_threshold = speed_threshold
        almost_inf = 9999
        high = np.array(
            [1, 1, 1, 1, 1, 1, 1, almost_inf, almost_inf, almost_inf], dtype=np.float32
        )
        low = -high
        if not VEL_STATE:
            high = high[0:7]
            low = low[0:7]

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if self.continuous:
            self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(7)

        self.reset()

    def _destroy(self):
        if not self.water:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.water)
        self.water = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.ship)
        self.ship = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.legs = []
        self.world.DestroyBody(self.containers[0])
        self.world.DestroyBody(self.containers[1])
        self.containers = []

    @staticmethod
    def compute_leg_length(LEG_LENGTH, level):
        if level > 2:
            return LEG_LENGTH * 0.1
        else:
            return LEG_LENGTH

    def reset(self, seed=None, options=None):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.episode_number += 1
        self.throttle = 0
        self.gimbal = 0.0
        self.total_landed_ticks += self.landed_ticks
        self.landed_ticks = 0
        self.stepnumber = 0

        self.terrainheigth = self.H / 20
        self.shipheight = self.terrainheigth + SHIP_HEIGHT
        ship_pos = self.W / 2
        self.helipad_x1 = ship_pos - SHIP_WIDTH / 2
        self.helipad_x2 = self.helipad_x1 + SHIP_WIDTH
        self.helipad_y = self.terrainheigth + SHIP_HEIGHT

        self.water = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=(
                        (-self.W, 0),
                        (2.0 * self.W, 0),
                        (2.0 * self.W, self.terrainheigth),
                        (-self.W, self.terrainheigth),
                    )
                ),
                friction=0.1,
                restitution=0.0,
            )
        )
        self.water.color1 = rgb(70, 96, 176)

        self.ship = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=(
                        (self.helipad_x1, self.terrainheigth),
                        (self.helipad_x2, self.terrainheigth),
                        (self.helipad_x2, self.terrainheigth + SHIP_HEIGHT),
                        (self.helipad_x1, self.terrainheigth + SHIP_HEIGHT),
                    )
                ),
                friction=0.5,
                restitution=0.0,
            )
        )

        self.containers = []
        for side in [-1, 1]:
            self.containers.append(
                self.world.CreateStaticBody(
                    fixtures=fixtureDef(
                        shape=polygonShape(
                            vertices=(
                                (
                                    ship_pos + side * 0.95 * SHIP_WIDTH / 2,
                                    self.helipad_y,
                                ),
                                (
                                    ship_pos + side * 0.95 * SHIP_WIDTH / 2,
                                    self.helipad_y + SHIP_HEIGHT,
                                ),
                                (
                                    ship_pos + side * 0.95 * SHIP_WIDTH / 2 - side * SHIP_HEIGHT,
                                    self.helipad_y + SHIP_HEIGHT,
                                ),
                                (
                                    ship_pos + side * 0.95 * SHIP_WIDTH / 2 - side * SHIP_HEIGHT,
                                    self.helipad_y,
                                ),
                            )
                        ),
                        friction=0.2,
                        restitution=0.0,
                    )
                )
            )
            self.containers[-1].color1 = rgb(206, 206, 2)

        self.ship.color1 = (0.2, 0.2, 0.2)

        def initial_rocket_pos(level):
            initial_x = self.W / 2
            initial_y = self.H * 0.95
            if level >= 2:
                initial_x_random = 0.3
            elif level == 1:
                initial_x_random = 0.1
            else:
                initial_x_random = 0.03
            initial_x += self.W * np.random.uniform(
                -initial_x_random, initial_x_random)
            return initial_x, initial_y

        initial_x, initial_y = initial_rocket_pos(self.level_number)
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=(
                        (-ROCKET_WIDTH / 2, 0),
                        (+ROCKET_WIDTH / 2, 0),
                        (+ROCKET_WIDTH / 2, +ROCKET_HEIGHT),
                        (-ROCKET_WIDTH / 2, +ROCKET_HEIGHT),
                    )
                ),
                density=1.0,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )

        self.lander.color1 = rgb(230, 230, 230)

        leg_length_modified = self.compute_leg_length(
            LEG_LENGTH, self.level_number)
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY,
                          initial_y + ROCKET_WIDTH * 0.2),
                angle=(i * BASE_ANGLE),
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=(
                            (0, 0),
                            (0, leg_length_modified / 25),
                            (i * leg_length_modified, 0),
                            (i * leg_length_modified, -leg_length_modified / 20),
                            (i * leg_length_modified / 3, -leg_length_modified / 7),
                        )
                    ),
                    density=1,
                    restitution=0.0,
                    friction=0.2,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (0.25, 0.25, 0.25)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(i * LEG_AWAY, ROCKET_WIDTH * 0.2),
                localAnchorB=(0, 0),
                enableLimit=True,
                maxMotorTorque=2500.0,
                motorSpeed=-0.05 * i,
                enableMotor=True,
            )
            djd = distanceJointDef(
                bodyA=self.lander,
                bodyB=leg,
                anchorA=(i * LEG_AWAY, ROCKET_HEIGHT / 8),
                anchorB=leg.fixtures[0].body.transform *
                (i * leg_length_modified, 0),
                collideConnected=False,
                frequencyHz=0.01,
                dampingRatio=0.9,
            )
            if i == 1:
                rjd.lowerAngle = -SPRING_ANGLE
                rjd.upperAngle = 0
            else:
                rjd.lowerAngle = 0
                rjd.upperAngle = +SPRING_ANGLE
            leg.joint = self.world.CreateJoint(rjd)
            leg.joint2 = self.world.CreateJoint(djd)

            self.legs.append(leg)

        def random_factor_velocity(level):
            if level >= 2:
                return 0.3
            elif level == 1:
                return 0.1
            else:
                return 0

        random_velocity_factor = random_factor_velocity(self.level_number)

        self.lander.linearVelocity = (
            -np.random.uniform(0, random_velocity_factor) *
            self.START_SPEED * (initial_x - self.W / 2) / self.W,
            -self.START_SPEED,
        )

        self.lander.angularVelocity = (1 + random_velocity_factor) * np.random.uniform(
            -1.0, 1.0
        )

        self.drawlist = (
            self.legs + [self.water] + [self.ship] +
            self.containers + [self.lander]
        )

        if self.continuous:
            obs, _, _, _, info = self.step([0, 0, 0])
            return obs, info
        else:
            obs, _, _, _, info = self.step(6)
            return obs, info

    def step(self, action):

        self.force_dir = 0

        if self.continuous:
            np.clip(action, -1, 1)
            self.gimbal += action[0] * 0.6 / FPS
            self.throttle += action[1] * 0.6 / FPS
            if action[2] > 0.5:
                self.force_dir = 1
            elif action[2] < -0.5:
                self.force_dir = -1
        else:
            if action == 0:
                self.gimbal += 0.01
            elif action == 1:
                self.gimbal -= 0.01
            elif action == 2:
                self.throttle += 0.01
            elif action == 3:
                self.throttle -= 0.01
            elif action == 4:  # left
                self.force_dir = -1
            elif action == 5:  # right
                self.force_dir = 1

        self.gimbal = np.clip(self.gimbal, -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.power = (
            0
            if self.throttle == 0.0
            else MIN_THROTTLE + self.throttle * (1 - MIN_THROTTLE)
        )

        # main engine force
        force_pos = (self.lander.position[0], self.lander.position[1])
        force = (
            -np.sin(self.lander.angle + self.gimbal) *
            MAIN_ENGINE_POWER * self.power,
            np.cos(self.lander.angle + self.gimbal) *
            MAIN_ENGINE_POWER * self.power,
        )
        self.lander.ApplyForce(force=force, point=force_pos, wake=False)

        # control thruster force
        force_pos_c = self.lander.position + THRUSTER_HEIGHT * np.array(
            (-np.sin(self.lander.angle), np.cos(self.lander.angle))
        )
        force_c = (
            -self.force_dir * np.cos(self.lander.angle) * SIDE_ENGINE_POWER,
            -self.force_dir * np.sin(self.lander.angle) * SIDE_ENGINE_POWER,
        )
        self.lander.ApplyLinearImpulse(
            impulse=force_c, point=force_pos_c, wake=False)

        self.world.Step(1.0 / FPS, 60, 60)

        pos = self.lander.position
        vel_l = np.array(self.lander.linearVelocity) / self.START_SPEED
        vel_a = self.lander.angularVelocity
        x_distance = 2.0 * (pos.x - self.W / 2) / self.W
        y_distance = 2.0 * ((pos.y - self.shipheight) /
                            (self.H - self.shipheight) - 0.5)

        angle = (self.lander.angle / np.pi) % 2
        # Normalize to [-1, 1]
        if angle > 1:
            angle -= 2

        state = [
            x_distance,
            y_distance,
            angle,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            2 * (self.throttle - 0.5),
            (self.gimbal / GIMBAL_THRESHOLD),
        ]

        if VEL_STATE:
            state.extend([vel_l[0], vel_l[1], vel_a])

        self.state = state

        # REWARD -------------------------------------------------------------------------------------------------------

        done = False
        reward = 0

        # fuel_cost = 0.1 * (0.0 * self.power + abs(self.force_dir)) / FPS
        # Too large => free-falling agent to save fuel (stop using all engine forces to save fuel).
        fuel_cost = 0.2 / FPS
        reward -= fuel_cost

        speed = np.linalg.norm((vel_l[0], vel_l[1]))

        info = {'is_success': False}

        outside = abs(pos.x - self.W / 2) > self.W / 2.0 or pos.y > self.H
        ground_contact = self.legs[0].ground_contact or self.legs[1].ground_contact
        broken_leg = (
            self.legs[0].joint.angle < -0.05 or self.legs[1].joint.angle > 0.05
        ) and ground_contact

        if outside:
            done = True
            info['is_success'] = False
            print('Outside!')
        elif self.game_over:
            done = True
            info['is_success'] = False
            # print('Crashed!')

        elif broken_leg:
            done = True
            info['is_success'] = False
            # print('Broken leg!')
        else:
            shaping = 0
            # The main engine force affects the orientation and angular velocity of the rocket.
            # If the penalty is too large, the agent is discouraged from using the main engine force.
            # As a consequence, the rocket will be free-falling (least changes in orientation and angular velocity).
            # If the penalty is too small, the agent is not centivised enough
            # to stabilize the orientation and reduce the angular velocity.
            # Encourage the rocket to quickly stabilize its orientation.
            shaping -= 0.5 * abs(angle)
            shaping -= 0.5 * abs(vel_a)

            # Encourage the rocket to quickly reduce its speed.
            # shaping -= 0.5 * speed
            shaping -= 0.5 * speed**2

            # Encourage the rocket to quickly navigate to the ship's center.
            shaping -= 3.0 * abs(x_distance)
            shaping -= 2.5 * (y_distance + 1.0) / 2.0

            # Reward for each leg touching the ground from a non-touching state in the previous step.
            shaping += 0.1 * \
                (self.legs[0].ground_contact + self.legs[1].ground_contact)

            landed = self.legs[0].ground_contact and self.legs[1].ground_contact and speed < 0.05

            if landed:
                self.landed_ticks += 1
            else:
                self.landed_ticks = 0
            shaping += self.landed_ticks / FPS

            if self.prev_shaping is not None:
                reward += shaping - self.prev_shaping
            self.prev_shaping = shaping

            if self.landed_ticks >= FPS:
                reward = 10.0
                done = True
                info['is_success'] = True
                print('Successful landing!')

        if done:
            reward += max(-3.0, -2.0 * (speed + abs(x_distance) +
                          y_distance + abs(angle) + abs(vel_a)))
        else:
            reward = np.clip(reward, -1, 1)

        # REWARD -------------------------------------------------------------------------------------------------------

        self.stepnumber += 1

        return np.array(state).astype(np.float32), reward, done, False, info

    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.box2d import rendering

        if self.viewer is None:

            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

            self.viewer.set_bounds(0, self.W, 0, self.H)

            sky = rendering.FilledPolygon(
                ((0, 0), (0, self.H), (self.W, self.H), (self.W, 0)))
            self.sky_color = rgb(126, 150, 233)
            sky.set_color(*self.sky_color)
            self.sky_color_half_transparent = (
                np.array((np.array(self.sky_color) + rgb(255, 255, 255))) / 2
            )
            self.viewer.add_geom(sky)

            self.rockettrans = rendering.Transform()

            engine = rendering.FilledPolygon(
                (
                    (0, 0),
                    (ENGINE_WIDTH / 2, -ENGINE_HEIGHT),
                    (-ENGINE_WIDTH / 2, -ENGINE_HEIGHT),
                )
            )
            self.enginetrans = rendering.Transform()
            engine.add_attr(self.enginetrans)
            engine.add_attr(self.rockettrans)
            engine.set_color(0.4, 0.4, 0.4)
            self.viewer.add_geom(engine)

            self.fire = rendering.FilledPolygon(
                (
                    (ENGINE_WIDTH * 0.4, 0),
                    (-ENGINE_WIDTH * 0.4, 0),
                    (-ENGINE_WIDTH * 1.2, -ENGINE_HEIGHT * 5),
                    (0, -ENGINE_HEIGHT * 8),
                    (ENGINE_WIDTH * 1.2, -ENGINE_HEIGHT * 5),
                )
            )
            self.fire.set_color(*rgb(255, 230, 107))
            self.firescale = rendering.Transform(scale=(1, 1))
            self.firetrans = rendering.Transform(
                translation=(0, -ENGINE_HEIGHT))
            self.fire.add_attr(self.firescale)
            self.fire.add_attr(self.firetrans)
            self.fire.add_attr(self.enginetrans)
            self.fire.add_attr(self.rockettrans)

            self.gridfins = []
            for i in (-1, 1):
                finpoly = (
                    (i * ROCKET_WIDTH * 1.1, THRUSTER_HEIGHT * 1.01),
                    (i * ROCKET_WIDTH * 0.4, THRUSTER_HEIGHT * 1.01),
                    (i * ROCKET_WIDTH * 0.4, THRUSTER_HEIGHT * 0.99),
                    (i * ROCKET_WIDTH * 1.1, THRUSTER_HEIGHT * 0.99),
                )
                gridfin = rendering.FilledPolygon(finpoly)
                gridfin.add_attr(self.rockettrans)
                gridfin.set_color(0.25, 0.25, 0.25)
                self.gridfins.append(gridfin)

        self.viewer.add_onetime(self.fire)
        for g in self.gridfins:
            self.viewer.add_onetime(g)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)

        for leg in zip(self.legs, [-1, 1]):
            path = [
                self.lander.fixtures[0].body.transform *
                (leg[1] * ROCKET_WIDTH / 2, ROCKET_HEIGHT / 8),
                leg[0].fixtures[0].body.transform *
                (leg[1] * self.compute_leg_length(LEG_LENGTH,
                 self.level_number) * 0.8, 0),
            ]
            self.viewer.draw_polyline(
                path, color=self.ship.color1, linewidth=1 if self.START_HEIGHT > 500 else 2
            )

        self.viewer.draw_polyline(
            (
                (self.helipad_x2, self.terrainheigth + SHIP_HEIGHT),
                (self.helipad_x1, self.terrainheigth + SHIP_HEIGHT),
            ),
            color=rgb(206, 206, 2),
            linewidth=1,
        )

        self.rockettrans.set_translation(*self.lander.position)
        self.rockettrans.set_rotation(self.lander.angle)
        self.enginetrans.set_rotation(self.gimbal)
        self.firescale.set_scale(
            newx=1, newy=self.power * np.random.uniform(1, 1.3))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        pass


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255
