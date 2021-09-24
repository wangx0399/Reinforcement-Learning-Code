import gym
import numpy as np
import torch
import torch.nn.functional as F
import os
import pandas as pd
from model import SAC_net
from utils.replay_buffer import ReplayBuffer

'''
Soft AC: a stochistic policy with max entropy, Q1 and Q2
        state --> mean and std, Normal(mean, std) sample action
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SAC(env_name='Hopper-v2',
        total_timesteps=int(1e6),
        start_timesteps=25000,
        train_timesteps=25000,
        batch_size=256
        ):

    env = gym.make(env_name)
    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print('gym_game: ', env_name, '\n state_dims: ', state_dims, 'action_dims: ', action_dims)

    Actic = SAC_net(state_dims, action_dims, max_action,
                    actor_lr=3e-4, critic_lr=3e-4, device=device, discount=0.995, tau=0.005)
    Memory = ReplayBuffer(state_dims, action_dims)
    Actic.train()
    global_info = {'global_len': [],
                   'global_return': []}
    whether_update_pi = 0
    episode_return = 0
    episode_len = 0
    max_episode_return = 0
    critic_loss, actor_loss, alpha_loss, alpha = 0, 0, 0, 0
    state = env.reset()
    for t in range(total_timesteps):

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = Actic.interact_action(state)

        next_state, reward, done, _ = env.step(action)
        not_done = 1. if done else 0.
        Memory.add(state, action, reward, next_state,not_done)
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

        if t > train_timesteps and t % 1 == 0:
            Actic.train()
            data = Memory.random_sample(batch_size)
            s = data[0].to(device)
            a = data[1].to(device)
            r = data[2].to(device)
            next_s = data[3].to(device)
            not_done = data[4].to(device)

            with torch.no_grad():
                alpha = torch.exp(Actic.log_alpha)
                next_a, log_p_nexta = Actic.get_log_p(next_s)
                target_Q1, target_Q2 = Actic.critic_target(next_s, next_a)
                target_Q = r + Actic.discount * not_done * \
                           (torch.min(target_Q1, target_Q2) - alpha * log_p_nexta)
            current_Q1, current_Q2 = Actic.critic(s, a)
            #print(alpha.shape, log_p_nexta.shape)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            Actic.critic_optim.zero_grad()
            critic_loss.backward()
            Actic.critic_optim.step()

            whether_update_pi += 1
            if whether_update_pi % 1 == 0:
                whether_update_pi = 0

                new_a, log_p_newa = Actic.get_log_p(s)
                target_net_Q1, target_net_Q2 = Actic.critic_target(s, new_a)
                Q = torch.min(target_net_Q1, target_net_Q2)
                actor_loss = (alpha * log_p_newa - Q).mean()
                Actic.actor_optim.zero_grad()
                actor_loss.backward()     #retain_graph=True
                Actic.actor_optim.step()

                target_entropy = - action_dims
                alpha_loss = ((- log_p_newa - target_entropy).detach() * Actic.log_alpha.exp()).mean()
                Actic.alpha_optim.zero_grad()
                alpha_loss.backward()
                Actic.alpha_optim.step()

                for param, target_param in zip(Actic.critic.parameters(), Actic.critic_target.parameters()):
                    target_param.data.copy_(Actic.tau * param.data + (1 - Actic.tau) * target_param.data)

        if t % 1000 == 0 and t > 5000:
            print('--- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ---')
            print('global timesteps {}  max episode return {:.2f}  20 mean episode length {:.0f}'
                  .format(t, max_episode_return, np.mean(global_info['global_len'][-20:])))

            print('critic loss {:.4f}      actor loss {:.4f}         20 mean episode return {:.2f}'
                  .format(critic_loss, actor_loss, np.mean(global_info['global_return'][-20:])))
            if t > train_timesteps:
                print('aplha loss: ', alpha_loss.detach().cpu(), '      alpha: ', alpha.cpu())


    return Actic, global_info


if __name__ == '__main__':

    RESULT_DIR = '/home/wangxu/Desktop/Reinforcement Learning Code/Soft_AC/result/'
    os.makedirs(RESULT_DIR, exist_ok=True)
    Mujoco = ['Hopper-v2']
    env_name = Mujoco[0]
    # Sample and Train
    actic, global_info = SAC(env_name='Hopper-v2',
                            total_timesteps=int(1e6),
                            start_timesteps=25000,
                            train_timesteps=25000,
                            batch_size=256)
    # Save nural network model parameters
    torch.save({'model_state_dict': actic.state_dict()},
               os.path.join(RESULT_DIR, 'SAC_model_parameters_{}.path'.format(env_name)))
    # Save train record
    record = pd.DataFrame(global_info)
    record.to_csv(os.path.join(RESULT_DIR, 'SAC_{}.csv'.format(env_name)))
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
            act = actic.interact_action(state)
            state, r, done, _ = env.step(act)
            reward += r
            i += 1
            if done or i == 2000:
                print('test {:.0f} time,  reward: {:.0f},  stop at {:.0f} step.'.format(t + 1, reward, i))
                env.close()
                break
                