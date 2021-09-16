import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

with open('statistics.pkl', 'rb') as f:
     data = pickle.load(f)['mean_episode_rewards']
num = len(data)
episode = np.linspace(0, num-1, num)

fig, ax = plt.subplots()
ax.plot(episode, data, color='b')
plt.show()