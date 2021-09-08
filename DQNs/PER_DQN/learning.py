import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.memory import Memory
from utils.gym import get_wrapper_by_name

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
    }


def per_dqn_learning(
        env,
        q_func,
        optimizer_spec,
        exploration,
        stopping_criterion=None,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        sample_times=1
        ):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape      # 84,84,1
        input_arg = frame_history_len * img_c                  # 4
    num_actions = env.action_space.n                           # 6 Pong

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0           # (1, 4, 84, 84)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                return model(obs.cuda()).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([random.randrange(num_actions)])

    # Initialize target q function and q function
    Q = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    memory = Memory(capacity=replay_buffer_size)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 1000

    for t in count():

        if stopping_criterion is not None and stopping_criterion(env):
            print("HHHHi, env's total step is ", get_wrapper_by_name(env, "Monitor").get_total_steps())
            break

        ### Step the env and store the transition
        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
        memory.tree.add_frame(last_obs)              # obs.shape = (84, 84, 1)
        # encode_recent_observation will take the latest observation
        # that you pushed into the buffer and compute the corresponding input
        # that should be given to a Q network by appending some previous frames.
        recent_observations = memory.tree.give_state()         # current input_obs 4*84*84
        # Choose action
        action = select_epilson_greedy_action(Q, recent_observations, t)[0]
        # Advance one step
        obs, reward, done, _ = env.step(action)
        # clip rewards between -1 and 1
        reward = max(-1.0, min(reward, 1.0))
        # Store other info in replay memory
        memory.tree.add_effect(action, reward, done)

        memory.store()
        # Resets the environment when reaching an episode boundary.
        if done:
            # log episode information
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            #print("episode_rewards_len: ", len(episode_rewards))
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-20:])
                Statistic["mean_episode_rewards"].append(mean_episode_reward)
            if len(episode_rewards) > 20:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
                Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

            # 注意：done=1对应的next_obs未被储存，而被reset代替
            obs = env.reset()
        last_obs = obs

        ### Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples for us to learn something useful
        # until then, the model will not be initialized and random actions should be taken
        if t > learning_starts and t % learning_freq == 0 and memory.tree.data.can_sample(batch_size):
            for _ in range(sample_times):
                # Use the replay buffer to sample a batch of transitions
                # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
                # in which case there is no Q-value at the next state; at the end of an
                # episode, only the current state reward contributes to the target
                tree_idx, ISweights, obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = memory.sample(batch_size)
                # Convert numpy nd_array to torch variables for calculation
                ISweights = torch.from_numpy(ISweights).type(dtype).squeeze().cuda()       # [32, 1]
                obs_batch = (torch.from_numpy(obs_batch).type(dtype) / 255.0).cuda()
                act_batch = (torch.from_numpy(act_batch).long()).cuda()
                rew_batch = torch.from_numpy(rew_batch).cuda()
                next_obs_batch = (torch.from_numpy(next_obs_batch).type(dtype) / 255.0).cuda()
                not_done_mask = torch.from_numpy(1 - done_mask).type(dtype).cuda()
                #print(ISweights.shape)

                # Compute current Q value, q_func takes only state and output value for every state-action pair
                # We choose Q based on action taken.
                #                      gather()  value (32, 6) 聚合  index (32, 1)
                current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze()
                # Double DQN
                # Detach variable from the current graph since we don't want gradients for next Q to propagated
                next_max_q_index = Q(next_obs_batch).detach().max(1)[1]
                next_max_q = target_Q(next_obs_batch).detach().gather(1, next_max_q_index.unsqueeze(1)).squeeze()
                # Compute the target of the current Q values
                target_Q_values = rew_batch + (gamma * not_done_mask * next_max_q)
                #loss = F.smooth_l1_loss(current_Q_values * ISweights, target_Q_values * ISweights)
                TD_error = abs(target_Q_values - current_Q_values).detach().cpu().numpy()
                # print(TD_error)
                memory.batch_update(tree_idx, TD_error)
                loss = torch.mean(ISweights * (target_Q_values - current_Q_values) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_r = loss.item()

                num_param_updates += 1
            # Periodically update the target network by Q network to target Q network
            # every 10000 times Q update, Q move to Q_target
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())


        # every 1000 t, print one time
        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep (count): %d" % (t,))
            print("mean reward (20 episodes): %f" % mean_episode_reward)
            print("best mean reward: %f" % best_mean_episode_reward)
            print("episodes: %d" % len(episode_rewards))
            print("exploration: %f" % exploration.value(t))
            print("loss:  %f" % loss_r)
            print('---------------------------------------')
            sys.stdout.flush()

        # Dump statistics to pickle
    with open('statistics.pkl', 'wb') as f:
        pickle.dump(Statistic, f)
        print("Saved to %s" % 'statistics.pkl')

    # SAVE Q NET MODEL
    env_name = 'BreakoutNoFrameskip-v4'
    RESULT_DIR = '/home/wangxu/Desktop/Reinforcement Learning Code/DQNs/PER_DQN/'
    torch.save({'model_state_dict': Q.state_dict()}, os.path.join(RESULT_DIR, 'model_parameters_{}.path'.format(env_name)))
