# Cheetah-Trainer

This is am implementation of the Automated Residual Reinforcement Learning algorithm.

## Requirement
```
- python3.7
- tensorflow
- tf2rl
- pybullet
- gym
- wandb
```
## How to use
* Basic Usage
```
python main.py --gait sine --policy TD3 --optimiser TBPSA
```
The `gait` argument represents the gait pattern; it can be line/sine/rose/triangle.
The `policy` argument determines the RL agent; it can be SAC/TD3
The `optimiser` argument chooses the parameter optimisers; it can be BO/CMA/TBPSA
* More Arguments
`state-mode`: This could change the state representation of the RL module.
`leg-action-mode`: This could change the action representation of the RL module.
`optimisation-mask`: This could change the parameter search space of the black-box optimiser.
`num-history-observation`: whether used stacked states as RL observation.

## Acknowledgment
** Parts of this implementation are based on [tf2rl](https://github.com/keiohta/tf2rl). **

