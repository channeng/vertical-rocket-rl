# Vertical Rocket Landing with Reinforcement learning

Initial Environment and rendering is forked from (https://github.com/EmbersArc), developed based on LunarLander environment by OpenAI.

Significant modifications were made to the initial box2d environment with respect to: 
- Rocket kinematics as described in the report
- Custom Reward shaping
- Introduced Curriculum learning

Repository requirements were updated and made to work with latest RL libraries.

Training, evaluation and replay scripts were written from scratch.

The agent is trained in **CONTINUOUS ACTION** mode.

### STATE VARIABLES  
The state consists of the following variables:
  * X position  
  * Y position  
  * Angle  
  * First Leg Ground Contact Indicator  
  * Second Leg Ground Contact Indicator  
  * Throttle  
  * Engine Gimbal  
  * X velocity  
  * Y velocity  
  * Angular Velocity  
  
All state variables are normalized for improved training.

### Continuous control inputs are:
* Gimbal (Left/Right)
* Throttle (Up/Down)
* Control Thruster (Left/Right)

We train a ***PPO*** agent using stable baselines 3.

# Setup

`pip install -r requirements.txt`

To test the environment: `python test_env.py`. You should see a rocket falling from the sky.

# Training

To reproduce the training for each stage of potential-based reward shaping as discussed in the report:
1. Stage 1: `python train.py --name=stage-1 --stage 1`
2. Stage 2: `python train.py --name=stage-2 --stage 2`
3. Stage 3: `python train.py --name=stage-3 --stage 3`
4. Stage 4: `python train.py --name=stage-4 --stage 4`

To train with curriculum learning, with wind effect and increased legs sensitivity:
`python train.py --name=curriculum-learning --use_curriculum`

Model will be saved to: `saves/ppo-<model_name>/best_model.zip`

## Tensorboard X 

In order to see logs of your train run , you can execute command below

`tensorboard --logdir=models/ --host localhost`

## How to replay with best model

To replay a single iteration of the trained agent in a given environment:
1. Stage 1: `python replay.py --name=stage-1`
2. Stage 2: `python replay.py --name=stage-2`
3. Stage 3: `python replay.py --name=stage-3`
4. Stage 4: `python replay.py --name=stage-4`

To extend the length of the episode, modify `max_episode_steps` in `gym/envs/__init__.py`.
