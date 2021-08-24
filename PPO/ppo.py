import gym
import torch
import numpy as np
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from model import AC, initialize_weights
from utils.Memory import Memory, DataSet, RandG
from utils.Noramlization import Normalization, State_norm
from utils.Schedule_coefficient import LinearSche

from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd


def ppo():
    env = gym.make('Hopper-v2')
    max_episode_steps = env._max_episode_steps
    # ** .unwrapper ** Important!!!
    env = gym.make('Hopper-v2').unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('gym_game:  Hopper-v2', '\n state_dim:', state_dim, 'action_dim:', action_dim)

    network = AC(state_dim, action_dim)
    initialize_weights(network, "orthogonal")
    pi_optimizer = optim.Adam(network.pi.parameters(), lr=1e-4)
    v_optimizer = optim.Adam(network.v.parameters(), lr=1e-4)
    lr = 3e-4
    clip = 0.2
    network.train()

    sliding_state = Normalization((state_dim,), clip=5.0)

    epoch_mean = {'Return': [], 'Len': []}
    global_steps = 0

    for epoch in range(1000):  # epochs

        # decrease PPO ratio's clip coefficient, following the number of epoch increase
        now_clip = round(LinearSche(clip, 0., epoch, 2000), 4)
        # decrease the learning rate
        now_lr = LinearSche(lr, 0., epoch, 2000)
        for g in pi_optimizer.param_groups:
            g['lr'] = round(now_lr, 6)
        for h in v_optimizer.param_groups:
            h['lr'] = round(now_lr, 6)

        memory = Memory()  # initial memory

        num_steps = 0
        reward_list = []
        len_list = []
        p = 0
        while num_steps < 4096:
            """
            collect N episodes, make (1000*(N-1) < 5000) and (1000*N > 5000)
            """
            state = env.reset()
            state = sliding_state(state)
            reward_sum = 0  # one trajectory's rewards
            for t in range(max_episode_steps):       # 1000 time-steps
                action, value, logprob = network.step(torch.as_tensor(state, dtype=torch.float32))
                action = action.numpy()
                logprob = logprob.numpy()
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                notdone = 0 if done else 1
                memory.push(state, value, action, logprob, notdone, reward)
                p = t+1      # count
                if done:
                    break
                state = next_state
                state = sliding_state(state)
            # do every trajectory over
            num_steps += p
            global_steps += p
            reward_list.append(reward_sum)
            len_list.append(p)

        epoch_mean['Return'].append(np.array(reward_list).mean())
        epoch_mean['Len'].append(np.array(len_list).mean())

        buffer_size = len(memory)
        print(buffer_size)
        states, values, actions, logp_olds, notdones, rewards = memory.sample()
        returns, GAEs = RandG(buffer_size, values, notdones, rewards, 0.995, 0.97)
        '''
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        network.to(device)
        s, v, a, logp_old, ret, gae = states.to(device), values.to(device), actions.to(device), \
                                      logp_olds.to(device), returns.to(device), GAEs.to(device)
        '''
        s, v, a, logp_old, ret, gae = states, values, actions, logp_olds, returns, GAEs
        dataset = DataSet(s, v, a, logp_old, ret, gae)
        batchloader = DataLoader(dataset=dataset, batch_size=512, shuffle=True, num_workers=4, drop_last=True)
        for _ in range(10):
            for step, data in enumerate(batchloader):
                mini_s, mini_v, mini_a, mini_logp_old, mini_ret, mini_gae = data

                mini_values = network.v(mini_s)
                #v_clip = mini_v + (mini_values - mini_v).clamp(- now_clip, now_clip)
                #loss_v1 = (mini_values - mini_ret).pow(2)
                #loss_v2 = (v_clip - mini_ret).pow(2)
                #loss_value = torch.mean(torch.min(loss_v1, loss_v2))
                v6std = 6 * mini_ret.std()
                loss_value = torch.mean((mini_values - mini_ret).pow(2)) / v6std
                v_optimizer.zero_grad()
                loss_value.backward()
                # gradient clip  max=5  by 2-norm
                #torch.nn.utils.clip_grad_norm_(network.v.parameters(), max_norm=5, norm_type=2)
                v_optimizer.step()

                _, mini_logp_new = network.pi(mini_s, mini_a)                                # [512,3]
                ratio = torch.exp(mini_logp_new - mini_logp_old)                             # [512,3]
                ratio_clip = ratio.clamp(1-now_clip, 1+now_clip)                             # [512,3]
                A = mini_gae.unsqueeze(1).expand_as(mini_a)                                  # [512] --> [512,3]
                loss_surr = torch.min(ratio * A, ratio_clip * A).sum(1)
                loss_entropy = - (torch.exp(mini_logp_new) * mini_logp_new).sum(1)
                loss_kl = (torch.exp(mini_logp_old) * (mini_logp_old - mini_logp_new)).sum(1)
                pi_optimizer.zero_grad()
                total_loss = torch.mean(- loss_surr - 0.01 * loss_entropy + 0.5 * loss_kl)
                total_loss.backward()
                #torch.nn.utils.clip_grad_norm_(network.pi.parameters(), max_norm=5, norm_type=2)
                pi_optimizer.step()
                #print(loss_kl)
        #network.cpu()

        print('epoch : {}   reward : {:.4f}   length : {:.4f}'
              .format(epoch, np.array(reward_list).mean(), np.array(len_list).mean()))
        print('pi_ADMA_lr: {}    v_ADMA_lr: {}     v_loss_mean: {:.6f}'
              .format(pi_optimizer.param_groups[0]['lr'], v_optimizer.param_groups[0]['lr'], loss_value))
        print('pi_loss_mean: {:.6f} = - {:.6f} - 0.01 * ({:.6f}) + 0.5 * ({:.6f})'
              .format(total_loss.mean(), loss_surr.mean(), loss_entropy.mean(), loss_kl.mean()))
        print('------------------------------------------------------------------------------')

    return network, sliding_state, epoch_mean


if __name__ == '__main__':
    RESULT_DIR = '/home/wangxu/Desktop/Reinforcement Learning Code/PPO/result/'
    mkdir(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #torch.manual_seed(1234)
    #torch.cuda.manual_seed(1234)
    #np.random.seed(1234)
    game = 'Hopper-v2'
    ACnet, sliding_state, epoch_mean = ppo()

    record = pd.DataFrame(epoch_mean)
    record.to_csv(joindir(RESULT_DIR, 'PPO_{}.csv'.format(game)))
    torch.save({
        'model_state_dict': ACnet.state_dict(),
        'normalized_state': sliding_state
    }, joindir(RESULT_DIR, 'PPO_model_parameters_{}.path'.format(game)))

    env = gym.make('Hopper-v2').unwrapped
    for t in range(5):
        obs = env.reset()
        obs = sliding_state(obs)
        done = False
        i = 0
        reward = 0
        while not done:
            env.render()
            act = ACnet.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, r, done, _ = env.step(act)
            obs = sliding_state(obs)
            reward += r
            i += 1
            if done or i == 2000:
                print('test {:.0f} time,  reward: {:.0f},  stop at {:.0f} step.'.format(t+1, reward, i))
                env.close()
                break
