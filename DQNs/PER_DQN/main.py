import gym
import torch.optim as optim
from model import DQN
from learning import OptimizerSpec, per_dqn_learning
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import ExpSchedule

"""
Summary:
    1. 要用 NoFrameskip 的版本
    2. atari_wrapper.py里的函数用于包装env=gym.make("...")，利于算法收敛
    3. PongNoFrameskip-v4 的 learning_freq 设为 1 ， 在迭代150000次开始收敛
    4. 稀疏奖励下，Loss function用smooth_L1_loss, Optim用RMSprop
    5. 每个episode,实际互动的次数是采样到的frame_skip倍
    6. “Monitor”可以用来监控env
"""


BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 262144  # 1000000
LEARNING_STARTS = 10000-1      # 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 1000
LEARNING_RATE = 0.0002
ALPHA = 0.95
EPS = 0.001


def main(env, train_timesteps):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        #print('--1--  ', env.get_total_steps())
        #print('--2--  ', get_wrapper_by_name(env, "Monitor").get_total_steps())
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= train_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = ExpSchedule(100000, 0.05)

    per_dqn_learning(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

if __name__ == '__main__':
    train_time_4 = 4 * 6000000
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env('BreakoutNoFrameskip-v4', seed)
    # print(env.observation_space.shape)      (84*84*1)
    #env.render()
    main(env, train_time_4)


    """
    MontezumaRevenge
    PongNoFrameskip-v4
    """