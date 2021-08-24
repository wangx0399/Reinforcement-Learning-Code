import gym
import time
from threading import Thread
from tkinter import *


class Game(Thread):
    def __init__(self):
        super(Game, self).__init__()
        self.env = gym.make('Breakout-v4')
        self.env.reset()
        self.action = 0
        self.total_reward = 0

    def run(self):
        while True:
            self.env.render()
            observation, reward, done, info = self.env.step(self.action)

            self.total_reward += reward
            if done:
                print("Episode finished")
                print(self.total_reward)
                break
            time.sleep(.05)


g = Game()
g.start()
print('start')

root = Tk()

def key(event):
    # print(event.char)
    key_map = {
        'w': 1,  # 发射
        's': 0,  # 停止
        'a': 3,  # 左
        'd': 2  # 右
    }
    if event.char != '':
        g.action = key_map[event.char]

    print(f'reword:{g.total_reward}')


frame = Frame(root, width=100, height=100)
frame.focus_set()
frame.bind("<Key>", key)
frame.pack()
root.mainloop()