import gym
import numpy as np
import torch
import os
import pandas as pd
from itertools import count
from model import TD3_net
from utils.action_noise import NormalNoise, OrnsteinUhlenbeckNoise
from utils.replay_buffer import ReplayBuffer

'''
DDPG, determinisitic policy
update sigma = r + gamma * Q_targ(s', Pi_targ(s'))
exploration noise: N(0, 0.1)
update noise: N(0, 0.2).clip(-0.5, 0.5)
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def TD3(env_name='Hopper-v2',
         total_timesteps=int(6e6),
         statr_timesteps=10000,
         train_timesteps=20000,
         batch_size=64):

    env = gym.make(env_name)
    #max_episode_steps = env._max_episode_step()
    #env = env.unwrapped
    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print('gym_game:  Hopper-v2', '\n state_dim:', state_dims, 'action_dim:', action_dims)

    Actic = TD3_net(state_dims, action_dims, device,  actor_lr=1e-3, critic_lr=1e-3)
    exploration_noise = NormalNoise(action_dims, mu=0., sigma=0.1)
    update_noise = NormalNoise(action_dims, mu=0., sigma=0.2, whether_tensor=True)
    update_noise_clip = 0.5
    Memory = ReplayBuffer(state_dims, action_dims)
    Actic.train()
    whether_update_pi = 0
    global_info = {'global_len': [],
                   'global_return': []}
    episode_return = 0
    episode_len = 0
    max_episode_return = 0
    done = False
    critic_loss, actor_loss = 0, 0
    state = env.reset()

    for t in range(total_timesteps):

        if t < statr_timesteps:
            action = env.action_space.sample()
        else:
            action = (Actic.select_action(state) + exploration_noise()).clip(-1., 1.)
            action = action * max_action

        next_state, reward, done, _ = env.step(action)
        not_done = 1. if done else 0.
        Memory.add(state, action, reward, next_state, not_done)
        episode_return += reward
        episode_len += 1

        if done:
            state = env.reset()
            max_episode_return = max(max_episode_return, episode_return)
            global_info['global_len'].append(episode_len)
            global_info['global_return'].append(episode_return)
            episode_return = 0
            episode_len = 0
        else:
            state = next_state

        if t > train_timesteps and t % 2 == 0:
            Actic.train()
            data = Memory.random_sample(batch_size)
            s = data[0].to(device)
            a = data[1].to(device)
            r = data[2].to(device)
            next_s = data[3].to(device)
            not_done = data[4].to(device)

            #with torch.no_grad():
            next_a = Actic.actor_target(next_s) + \
                     torch.clamp(update_noise(), -update_noise_clip, update_noise_clip).to(device)
            next_a = torch.clamp(next_a, -1., 1.) * max_action
            next_Q1, next_Q2 = Actic.critic_target(next_s, next_a)
            target_Q = r + (not_done * Actic.discount * torch.min(next_Q1, next_Q2)).detach()
            current_Q1, current_Q2 = Actic.critic(s, a)
            critic1_loss = torch.nn.functional.mse_loss(current_Q1, target_Q)
            critic2_loss = torch.nn.functional.mse_loss(current_Q2, target_Q)
            critic_loss = critic1_loss + critic2_loss

            Actic.critic_optim.zero_grad()
            critic_loss.backward()
            Actic.critic_optim.step()

            whether_update_pi += 1
            if whether_update_pi % 2 == 0:
                whether_update_pi = 0

                actor_loss = - Actic.critic.q_update_pi(s, Actic.actor(s)).mean()

                Actic.actor_optim.zero_grad()
                actor_loss.backward()
                Actic.actor_optim.step()

                for param, target_param in zip(Actic.critic.parameters(), Actic.critic_target.parameters()):
                    target_param.data.copy_(Actic.tau * param.data + (1 - Actic.tau) * target_param.data)

                for param, target_param in zip(Actic.actor.parameters(), Actic.actor_target.parameters()):
                    target_param.data.copy_(Actic.tau * param.data + (1 - Actic.tau) * target_param.data)


        if t % 1000 == 0 and t > 5000:
            print('--- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ---')
            print('global timesteps {}  max episode return {:.2f}  20 mean episode length {:.0f}'
                  .format(t, max_episode_return, np.mean(global_info['global_len'][-20:])))

            print('critic loss {:.4f}      actor loss {:.4f}         20 mean episode return {:.2f}'
                  .format(critic_loss, actor_loss, np.mean(global_info['global_return'][-20:])))

    return Actic, global_info

if __name__ == '__main__':

    RESULT_DIR = '/home/wangxu/Desktop/Reinforcement Learning Code/TD3/result/'
    os.makedirs(RESULT_DIR, exist_ok=True)
    Mujoco = ['Hopper-v2']
    env_name = Mujoco[0]
    # Sample and Train
    actic, global_info = TD3(env_name='Hopper-v2',
                            total_timesteps=int(1e6),
                            statr_timesteps=25000,
                            train_timesteps=25000,
                            batch_size=100)
    # Save nural network model parameters
    torch.save({'model_state_dict': actic.state_dict()},
               os.path.join(RESULT_DIR, 'DDPG_model_parameters_{}'.format(env_name)))
    # Save train record
    record = pd.DataFrame(global_info)
    record.to_csv(os.path.join(RESULT_DIR, 'PPO_{}.csv'.format(env_name)))
    # test
    env = gym.make(env_name).unwrapped
    max_action = float(env.action_space.high[0])
    for t in range(5):
        state = env.reset()
        done = False
        i = 0
        reward = 0
        while not done:
            env.render()
            act = actic.select_action(state) * max_action
            state, r, done, _ = env.step(act)
            reward += r
            i += 1
            if done or i == 2000:
                print('test {:.0f} time,  reward: {:.0f},  stop at {:.0f} step.'.format(t + 1, reward, i))
                env.close()
                break
