import logging
import multiprocessing
import threading
import os
import numpy as np
import tensorflow as tf

import Utils.utils
from Environment.environment import UnoEnvironment
from Environment.state import State
from Players.A3CPlayer import A3CPlayer
from Players.AgentPlaceholder import AgentPlaceholder
from Players.randomplayer import RandomPlayer
from Agents.ListMemory import ListMemory
from Agents.ActorCriticModel import ActorCriticModel

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

# Prirejeno po zgledu: https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras-eager-execution.html
class A3CWorker(threading.Thread):
    GLOBAL_MAX_EPISODES = 1000
    global_model_update_frequency = 20
    global_best_episode_score = None
    global_total_episodes_across_all_workers = 0
    global_semaphore = threading.Lock()
    global_statistic = []

    def __init__(self, master_model, optimizer, worker_id, folder, beta, gamma, opponent_model_path=None):
        super(A3CWorker, self).__init__()
        self.master_model = master_model
        self.optimizer = optimizer
        self.worker_id = worker_id
        self.env = UnoEnvironment(False)
        self.state_size = State.STATE_SIZE
        self.action_size = State.ACTION_SIZE
        self.gamma = gamma  # discounting factor
        self.beta = beta
        self.worker_model = ActorCriticModel(self.action_size)
        self.memory = ListMemory()
        self.folder = folder
        self.episode_loss = 0
        self.episode_steps = 0
        self.episode_reward = 0
        self.episode_discounted_reward = 0
        self.steps = 0
        self.save = True
        self.agent_placeholder = AgentPlaceholder()
        if opponent_model_path is not None:
            self.opponent = A3CPlayer(opponent_model_path)
        else:
            self.opponent = RandomPlayer()

    def run(self):
        while A3CWorker.global_total_episodes_across_all_workers < A3CWorker.GLOBAL_MAX_EPISODES:
            A3CWorker.global_total_episodes_across_all_workers += 1
            logger.info("Starting episode " + str(A3CWorker.global_total_episodes_across_all_workers) + "/" +
                        str(A3CWorker.GLOBAL_MAX_EPISODES) + " with worker " + str(self.worker_id))

            done = False
            self.episode_steps = 0
            self.episode_loss = 0
            self.episode_reward = 0
            self.episode_discounted_reward = 0
            self.memory.empty()
            state = self.env.reset(self.agent_placeholder, self.opponent)

            while not done:
                action = self.get_action()
                next_state, reward, done = self.env.step_with_opp_step(action)
                self.episode_reward += reward
                self.memory.store(state, action, reward, next_state, done)
                state = next_state

                self.steps += 1
                self.episode_steps += 1

                if self.steps % A3CWorker.global_model_update_frequency == 0 or done:
                    self.sync_gradients()

                if done:
                    A3CWorker.global_statistic.append((self.episode_reward, self.episode_loss / self.episode_steps))
                    if A3CWorker.global_best_episode_score is None or \
                            self.episode_reward > A3CWorker.global_best_episode_score:
                        self.update_master_model()

    def get_action(self):
        # You are only interested in the legal policy logits
        policy_logits, values = self.worker_model(tf.convert_to_tensor(np.random.random((1, self.state_size)),
                                                                       dtype=tf.float32))
        legal_actions = self.env.get_legal_actions()
        legal_policy_logits = np.take(policy_logits, legal_actions)
        stochastic_legal_action_probabilities = tf.nn.softmax(legal_policy_logits)
        stochastic_legal_action_probabilities = stochastic_legal_action_probabilities.numpy()
        stochastic_legal_action_probabilities /= stochastic_legal_action_probabilities.sum()
        stochastic_policy_driven_legal_action = np.random.choice(legal_actions,
                                                                 p=stochastic_legal_action_probabilities)
        return stochastic_policy_driven_legal_action

    def update_master_model(self):
        A3CWorker.global_best_episode_score = self.episode_reward
        if self.save:
            with A3CWorker.global_semaphore:
                logger.info("Saving best model "
                            "- worker: " + str(self.worker_id) +
                            " episode: " + str(A3CWorker.global_total_episodes_across_all_workers) +
                            " reward: " + str(self.episode_reward) +
                            " discounted reward:" + str(self.episode_discounted_reward) +
                            " loss: " + str(self.episode_loss.numpy()))
                self.master_model.save_weights(os.path.join(self.folder, 'model_' + Utils.utils.get_timestamp() +'.h5'))

    def sync_gradients(self):
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss()
        self.episode_loss += total_loss
        grads = tape.gradient(total_loss, self.worker_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.master_model.trainable_weights))

        self.worker_model.set_weights(self.master_model.get_weights())
        self.memory.empty()

    def compute_loss(self):
        if self.memory.dones[-1]:
            reward_sum = 0.
        else:
            reward_sum = self.worker_model(tf.convert_to_tensor(self.memory.next_states[-1][None, :], dtype=tf.float32))[-1].numpy()[0]
        discounted_rewards = []
        for reward in self.memory.rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        self.episode_discounted_reward = discounted_rewards[0]

        logits, values = self.worker_model(tf.convert_to_tensor(np.vstack(self.memory.states), dtype=tf.float32))

        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values # Baseline
        value_loss = advantage ** 2

        actions_one_hot = tf.one_hot(self.memory.actions, self.action_size, dtype=tf.float32)

        policy = tf.nn.softmax(logits)
        entropy = tf.reduce_sum(policy * tf.math.log(policy + self.beta), axis=1)

        # Entropija prepreč prehitro konvergiranje k suboptimalnem determinističnmem pravilniku
        policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=actions_one_hot, logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss

