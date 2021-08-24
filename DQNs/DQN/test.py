import gym
import numpy as np
from gym import wrappers
from utils.atari_wrapper import wrap_deepmind, NoopResetEnv
from utils.gym import get_wrapper_by_name, get_env

"""
env = gym.make('PongNoFrameskip-v4')

print(env._max_episode_steps)
print(env.observation_space.hape)
env4 = wrap_deepmind(env)
print(env4.observation_space.shape)
print('reset', env4.reset().shape)

env = NoopResetEnv(env)
print(env.reset().shape)
"""

env = gym.make('PongNoFrameskip-v4')
print(env)
print(env.__class__.__name__)

expt_dir = 'tmp/gym-results'
env1 = wrappers.Monitor(env, expt_dir, force=True)
print(env1)
print(env1.__class__.__name__)

env2 = wrap_deepmind(env1)
print(env2)
print(env2.__class__.__name__)
print(get_wrapper_by_name(env2, "Monitor"))




