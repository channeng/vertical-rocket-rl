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


# Training

`python ppo_rocket.py --name test`
- Don't forget to change training run each time you execute code or otherwise it will overwrite of older files.

Model will be saved to: `saves/ppo-test/\*.dat`

## Tensorboard X 

In order to see logs of your train run , you can execute command below

`tensorboard --logdir runs/ --host localhost`

## How to test 

Using a saved pre-trained model, run the command: 
`python rocket_play.py --model rocket_saved_network/PPO/actorbest_-2.977_1700000.dat`

To extend the length of the episode, modify `max_episode_steps=1000` in `gym/envs/__init__.py`. 

You should be able to see the rocket land successfully with `max_episode_steps=5000`.
