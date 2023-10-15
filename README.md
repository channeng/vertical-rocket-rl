# Vertical Rocket Landing with Reinforcement learning

Environment is created by [Sven Niederberger](https://github.com/EmbersArc), based on LunarLander environment by OpenAI.

It is possible to train in **DISCRETE ACTION**  or **CONTINUOUS ACTION** modes.
You can find more information at (gym/env/box2d/rocket_lander.py)

### STATE VARIABLES  
The state consists of the following variables:
  * X position  
  * Y position  
  * Angle  
  * First Leg Ground Contact Indicator  
  * Second Leg Ground Contact Indicator  
  * Throttle  
  * Engine Gimbal  
  
If VEL_STATE is set to true, the velocities are included:  
  * X velocity  
  * Y velocity  
  * Angular Velocity  
  
All state variables are normalized for improved training.
    

### Discrete control inputs are:
* Gimbal Left
* Gimbal Right
* Throttle Up
* Throttle Down
* Use First Control Thruster
* Use Second Control Thruster
* No Action

### Continuous control inputs are:
* Gimbal (Left/Right)
* Throttle (Up/Down)
* Control Thruster (Left/Right)


A ***PPO*** agent is included. Agent creation is done using [PTan Agent Network Library](https://github.com/Shmuma/ptan).


# Setup

`pip install -r requirements.txt`

To test: `python test_env.py`, to see a rocket falling.

# Environment

To simplify training, I've set the environment variables as:
- `gym/envs/box2d/vertical_rocket.py`
  - CONTINUOUS = False   --> DISCRETE action space
  - START_HEIGHT = 500.0
  - START_SPEED = 25.0
- `gym/envs/__init__.py`
  - max_episode_steps=500

In the original environment, the default environment variables are:
- `gym/envs/box2d/vertical_rocket.py`
  - CONTINUOUS = True   --> CONTINUOUS action space
  - START_HEIGHT = 1000.0
  - START_SPEED = 100.0
- `gym/envs/__init__.py`
  - max_episode_steps=1000

To extend the length of the episode, modify `max_episode_steps=1000` in `gym/envs/__init__.py`.


# Training

`python ppo_rocket_sb3.py --name=continuous1mil_new --method=ppo`
- Don't forget to change training run each time you execute code or otherwise it will overwrite of older files.

Model will be saved to: `saves/ppo-<model_name>/best_model.zip`

## Tensorboard X 

In order to see logs of your train run , you can execute command below

`tensorboard --logdir=models/ --host localhost`

## How to replay with best model

Using a saved pre-trained model, run the command: 
`python rocket_play_sb3.py --name=continuous1mil_new`

To extend the length of the episode, modify `max_episode_steps` in `gym/envs/__init__.py`.
