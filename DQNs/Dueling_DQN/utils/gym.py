import gym
from gym import wrappers
from utils.seed import set_global_seeds
from utils.atari_wrapper import wrap_deepmind


def get_env(env_name, seed):
    '''
    查看Monitor类的源代码：https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py，
    其中有一个video_callable变量，它在_start函数中有注释，该变量为Optional[function, False]，
    即我们要么传入一个函数，要么就传入False，默认是None，表示传入函数，但是函数是None。
    如果我们想要按照自己的喜好记录视频，那么写一个函数传给video_callable即可，
    比如说env=wrappers.Monitor(env,'/tmp/cartpole-experiment-1',video_callable=fun_udef)，
    fun_udef是我们自己定义的一个以episode index为输入，bool为输出的函数，如果某个episode，
    比如第一个episode输出为False，那么就不记录。
    '''
    env = gym.make(env_name)

    set_global_seeds(seed)
    # env set fixed seed, bad for agent to learn under various state
    #env.seed(seed)

    expt_dir = 'tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True, video_callable=None)
    env = wrap_deepmind(env)
    return env


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)