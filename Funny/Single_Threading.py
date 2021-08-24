import gym
import time
from threading import Thread


class Game(Thread):
    def __init__(self):
        super(Game, self).__init__()
        self.env = gym.make('Breakout-v4')
        self.env.reset()
        self.action = 0

    def run(self):
        while True:
            self.env.render()
            observation, reward, done, info = self.env.step(self.action)
            if done:
                print("Episode finished")
                break
            time.sleep(.2)


g = Game()
g.start()
print('start')

while True:
    s = input('action:\n')
    if s != '':
        print(s, int(s))
        g.action = int(s)