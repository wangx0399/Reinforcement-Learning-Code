import numpy as np
import torch
import gym
import os
from model import DQN
from gym import wrappers
from utils.seed import set_global_seeds
from utils.atari_wrapper import wrap_deepmind
import matplotlib.pyplot as plt


Atari_Envs = ['PongNoFrameskip-v4',
              'BreakoutNoFrameskip-v4']
atari_env = Atari_Envs[1]

seed = 1234 # Use a seed of zero (you may want to randomize the seed!)
env = gym.make(atari_env)

set_global_seeds(seed)
env.seed(seed)
expt_dir = 'tmp/gym-test'
env = wrappers.Monitor(env, expt_dir, force=True)
env = wrap_deepmind(env)
#plt.imshow(env.reset().reshape(84,84))
#plt.show()
frame_history_len = 4

img_h, img_w, img_c = env.observation_space.shape  # 84,84,1
input_arg = frame_history_len * img_c  # 4
input_shape = [input_arg, img_h, img_w]
num_actions = env.action_space.n  # 6 Pong


Q_net = DQN(input_arg, num_actions)
Q_net.eval()

RESULT_DIR = '/home/wangxu/Desktop/Reinforcement Learning Code/DQNs/PER_DQN/'
Parameters = torch.load(os.path.join(RESULT_DIR, 'model_parameters_{}.path'.format(atari_env)))

Q_net.load_state_dict(Parameters['model_state_dict'])

obs = torch.zeros(input_shape)
done = False
live = 1
Return = 0
i = 0
state = env.reset()

while 1:
    env.render()
    obs1 = obs[1, :, :]
    obs2 = obs[2, :, :]
    obs3 = obs[3, :, :]
    obs4 = torch.from_numpy((state / 255.0).astype(np.float32)).squeeze()
    obs = torch.stack((obs1, obs2, obs3, obs4), 0)#.permute(2, 0, 1)
    act = Q_net(obs.unsqueeze(0)).data.max(1)[1]
    state, reward, done, _ = env.step(act.numpy())
    Return += reward
    i += 1
    print(state.mean(), reward, act.numpy(), done)
    if done:
        live += 1
        state = env.reset()
        if live > 5:
            env.close()
            break
print("Return: {:.4f}".format(Return))

