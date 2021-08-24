import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M')


def plot(rewards, losses, algorithm, rewards2=None):
    folder = "plots\\" + algorithm
    if not os.path.exists(folder):
        os.makedirs(folder)

    rewards = np.clip(rewards, -100, 100)
    losses = np.clip(losses, -100, 100)
    if algorithm == "DQN":
        line_graph(np.arange(0, len(losses)), losses, "Batch", "Loss", algorithm, folder)
    else:
        # A3c
        line_graph(np.arange(0, len(losses)), losses, "Episode", "Average Loss", algorithm, folder)
    line_graph(np.arange(0, len(rewards)), rewards, "Episode", "Score", algorithm, folder)

    average_rewards = []
    for i in range(len(rewards) // 100):
        average_rewards.append(np.mean(rewards[100 * i:100 * (i + 1)]))

    line_graph(get_episodes_array_for_averages(average_rewards), average_rewards, "Episodes", "Average reward", algorithm, folder)

    if rewards2 is not None:
        rewards2 = np.clip(rewards2, -100, 100)
        line_graph(np.arange(0, len(rewards2)), rewards2, "Episode", "Score (p2)", algorithm + "(Other NFSP player)", folder)
        average_rewards = []
        for i in range(len(rewards2) // 100):
            average_rewards.append(np.mean(rewards2[100 * i:100 * (i + 1)]))

        line_graph(np.arange(0, get_episodes_array_for_averages(average_rewards)), average_rewards, "Episodes",
                   "Average reward (p2)", algorithm + "(Other NFSP player)", folder)


def get_episodes_array_for_averages(average_values):
    array = np.arange(1, len(average_values)+1) * 100
    return array


def line_graph(x, y, xlabel, ylabel, title, folder, color="blue"):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, color=color)
    #plt.show()
    plt.savefig(folder + '/' + ylabel + '_' + get_timestamp() +'.png')
    plt.show()
