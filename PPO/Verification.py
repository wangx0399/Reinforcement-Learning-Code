import torch
import gym
from gym import wrappers
from os.path import join
from os import makedirs
from model import AC
from PIL import Image
import shutil


MuJoCo_envs = ['Walker2d-v2',
               'Ant-v2',
               'HalfCheetah-v2',
               'Hopper-v2',
               'Swimmer-v2',
               'Humanoid-v2',
               'HumanoidStandup-v2']

########### START ############
MuJoCo_env = MuJoCo_envs[6]  #
# Save or not?               #
save_img = bool(0)           #
##############################

env = gym.make(MuJoCo_env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_gain = env.action_space.high.max()   # ******
ActorCritic = AC(state_dim, action_dim)
ActorCritic.eval()

PATH = '/home/wangxu/Desktop/Reinforcement Learning Code/PPO/result'
Parameters = torch.load(join(PATH, 'PPO_model_parameters_{}.path'.format(MuJoCo_env)))
# # # model.keys() : dict_keys(['model_state_dict', 'normalized_state'])

ActorCritic.load_state_dict(Parameters['model_state_dict'])
Normalized_state = Parameters['normalized_state']

# env = env.unwrapped
expt_dir = 'tmp/frame_pics/'
if save_img:
    shutil.rmtree(expt_dir)
    makedirs(expt_dir, exist_ok=True)

for t in range(1):
    obs = env.reset()
    obs = Normalized_state(obs)
    done = False
    i = 0
    reward = 0
    env.render()
    env.close()
    while not done:
        # env.render()
        if save_img:
            img = env.render(mode='rgb_array')
            img = Image.fromarray(img)
            img.save(join(expt_dir, '{}.jpg'.format(i)))
        else: env.render()
        act = ActorCritic.act(torch.as_tensor(obs, dtype=torch.float32))
        obs, r, done, _ = env.step(act * action_gain)
        obs = Normalized_state(obs)
        reward += r
        i += 1
        if done or i == 2000:
            print('test {:.0f} time,  reward: {:.0f},  stop at {:.0f} step.'.format(t+1, reward, i))
            env.close()
            break

