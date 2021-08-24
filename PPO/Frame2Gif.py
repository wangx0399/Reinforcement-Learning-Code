import imageio
import os


def create_gif(frame_dir, gif_dir, duration=0.02):
    frame_num = len(os.listdir(frame_dir))
    frames = []
    for i in range(800):  # frame_num
        img = os.path.join(frame_dir, '{}.jpg'.format(i))
        print(img)
        frames.append(imageio.imread(img))
    imageio.mimsave(gif_dir, frames, 'GIF', duration=duration)
    return


def main(env_name):
    frame_dir = '/home/wangxu/Desktop/Reinforcement Learning Code/PPO/tmp/frame_pics'
    gif_dir = '/home/wangxu/Desktop/Reinforcement Learning Code/PPO/tmp/' + env_name + '.gif'
    duration = 0.01
    create_gif(frame_dir, gif_dir, duration)


if __name__ == '__main__':
    MuJoCo_envs = ['Walker2d-v2',
                   'Ant-v2',
                   'HalfCheetah-v2',
                   'Hopper-v2',
                   'Swimmer-v2',
                   'Humanoid-v2',
                   'HumanoidStandup-v2']
    MuJoCo_env = MuJoCo_envs[3]
    main(MuJoCo_env)
