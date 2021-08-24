import multiprocessing
import os
import numpy as np
import tensorflow as tf

import Utils.utils
from Agents.ActorCriticModel import ActorCriticModel
from Environment.player import Player
from Environment.state import State
from Agents.A3CWorker import A3CWorker

# Prirejeno po zgledu: https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras-eager-execution.html

class A3CMaster(Player):
    def __init__(self, trained_model_path=None, opponent_model_path=None):
        self.learning_rate = 5e-4
        self.beta = 1e-35
        self.gamma = 0.99
        self.state_size = State.STATE_SIZE
        self.action_size = State.ACTION_SIZE
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.master_model = ActorCriticModel(self.action_size)  # global network
        self.master_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        if trained_model_path is not None:
            self.master_model.load_weights(trained_model_path)

        self.opponent_model_path = opponent_model_path
        self.folder = "models\\A3C"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.wins = 0

    def train(self):
        a3c_workers = [A3CWorker(self.master_model, self.optimizer, worker_id, self.folder,
                                 self.beta, self.gamma, opponent_model_path=self.opponent_model_path)
                       for worker_id in range(multiprocessing.cpu_count())]
        for i, worker in enumerate(a3c_workers):
            worker.start()
        [worker.join() for worker in a3c_workers]
        self.plot()

    def plot(self):
        rewards = []
        episode_average_losses = []
        for statistic in A3CWorker.global_statistic:
            reward, episode_average_loss = statistic
            rewards.append(reward)
            episode_average_losses.append(episode_average_loss)
        rewards = np.array(rewards)
        episode_average_losses = np.array(episode_average_losses)
        Utils.utils.plot(rewards, episode_average_losses, "A3C")

    def get_action(self, legal_actions, state):
        policy, value = self.master_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        mask = np.full(self.action_size, True)
        mask[legal_actions] = False
        policy = np.ma.array(policy, mask=mask)
        action = np.argmax(policy)
        return action


if __name__ == "__main__":
    agent = A3CMaster()
    agent.train()
