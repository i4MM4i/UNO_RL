import os
import warnings

import numpy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber, Reduction
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tqdm import tqdm

from Agents.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from Agents.SupervisedMemory import SupervisedMemory
from Environment.environment import UnoEnvironment
from Environment.player import Player
from Environment.state import State
from Players.AgentPlaceholder import AgentPlaceholder
from Utils import utils


class DeepQNetworkAgent(Player):

    def __init__(self, alpha=0.031157,
                 delta=0.13907,
                 epsilon_decay=0.99997,
                 eta=0.044575,
                 gamma=0.013082,
                 learning_rate=0.050023):
        super().__init__("DQN")
        self.action_size = State.ACTION_SIZE
        self.state_size = State.STATE_SIZE
        self.memory_rl = PrioritizedReplayBuffer(2000000)
        self.memory_sl = SupervisedMemory(2000000)
        self.batch_size = 512
        self.model_update_frequency = 10
        self.model_save_frequency = 100
        self.alpha = alpha   # Pred opt: 0.7
        self.delta = delta  # Pred opt: 0.5
        self.epsilon = 1
        self.epsilon_min = 0.001
        self.epsilon_decay = epsilon_decay  # Pred opt: 0.99999
        self.gamma = gamma  # 0 # 0.029559  # Pred opt: 0.01
        self.learning_rate = learning_rate  # Pred opt: 0.1
        self.learning_rate_sl = 0.005
        self.eta = eta  # Pred opt: 0.1
        self.number_of_episodes = 10000
        self.reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0)
        self.policy_network = self.build_model(self.learning_rate, 'linear',
                                               Huber(reduction=Reduction.SUM, delta=self.delta))
        self.target_network = self.build_model(self.learning_rate, 'linear',
                                               Huber(reduction=Reduction.SUM, delta=self.delta))
        self.target_network.set_weights(self.policy_network.get_weights())
        self.supervised_learning_network = self.build_model(self.learning_rate_sl, 'softmax',
                                                            tf.keras.losses.sparse_categorical_crossentropy)
        self.total_rewards_p1 = []
        self.total_rewards_p2 = []
        self.losses = []
        self.steps = 0
        self.p2 = AgentPlaceholder()
        self.rounds = 0
        self.n_batches = 0
        self.save_model = True
        self.env = UnoEnvironment(False)

    def build_model(self, learning_rate, output_activation, loss):
        model = Sequential()
        model.add(Dense(140, input_shape=(self.state_size,), activation='relu'))
        # model.add(Dense(190, activation='relu'))
        model.add(Dense(self.action_size, activation=output_activation))
        model.compile(loss=loss,
                      optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
        return model

    def predict(self, state):
        # TODO: prever kaj je fora, da je to tko spisan
        return self.policy_network.predict(np.array(state).reshape(-1, *state.shape))[0]

    def get_epsilon_greedy_action(self, legal_actions, state):
        # Z verjetnostjo v vrednosti epsilon, se iz izbere naključna poteza (exploration)
        # Drugače se izbere najboljša (greedy)
        if np.random.sample() < self.epsilon:
            # Izbere naključno legalno potezo
            action = np.random.choice(legal_actions, 1)
        else:
            # Poteza se vedno izbere s pomočjo nevronske mreže pravilnika (policy network)
            values = self.predict(state)
            mask = np.full(self.action_size, True)
            mask[legal_actions] = False
            # Zamaskira vse nelegalne poteze
            values = np.ma.array(values, mask=mask)
            # Pridobi potezeo z največjo predvideno vrednostjo
            action = np.argmax(values)
        return action.item()

    def get_average_action(self, legal_actions, state):
        # Selecting an action is always done by the policy network
        state = (np.array(state).reshape(-1, *state.shape))
        policy_logits = self.supervised_learning_network.predict(state)
        legal_policy_logits = np.take(policy_logits, legal_actions)
        stochastic_legal_action_probabilities = tf.nn.softmax(legal_policy_logits).numpy()
        stochastic_legal_action = np.random.choice(legal_actions, p=stochastic_legal_action_probabilities)
        return stochastic_legal_action

    def get_action(self, legal_actions, state):
        legal_actions = numpy.asarray(legal_actions)

        if np.random.sample() < self.eta:
            # average strategy
            action = self.get_average_action(legal_actions, state)
            # print("Average:", action)
            return action
        else:
            # epsilon greedy
            # add to supervised learning memory (doesn't make sense to add those which were made with the sl memory already)
            action = self.get_epsilon_greedy_action(legal_actions, state)
            self.memory_sl.add(state, action)
            # print("E-Greedy:", action)
            return action

    def train(self):
        # if self.steps % self.batch_size == 0:
        if len(self.memory_rl) > self.batch_size and len(self.memory_sl) > self.batch_size:
            self.reinforcement_learning()
            self.supervised_learning()
            self.reinforcement_learning()
            self.supervised_learning()
            self.n_batches += 1
        self.update_network()

    def update_network(self):
        if self.n_batches % self.model_update_frequency == 0:
            self.target_network.set_weights(self.policy_network.get_weights())

    def play(self, train=True):
        done = False
        self.env.reset(self, self.p2)
        p2_action = None
        p1_episode_reward = 0
        p2_episode_reward = 0
        while not done:
            p1_state = self.env.get_state(0)
            while self.env.current_player_index == 0 and not done:
                p1_action = self.get_action(self.env.get_legal_actions(), p1_state)
                # print("P1 will take:", p1_action)
                p1_next_state, done = self.env.step(p1_action)
                self.steps += 1
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                if self.env.current_player_index == 1:
                    if p2_action is not None:
                        p2_reward = self.env.calculate_built_in_reward(1)
                        p2_episode_reward += p2_reward
                        # print("Adding p2: " + str(p2_action))
                        # print("Reward for p2:" + str(p2_reward))
                        p2_next_state = self.env.get_state(1)
                        self.memory_rl.add(p2_state, p2_action, p2_reward, p2_next_state, done)
                elif done:
                    if p2_action is not None:
                        p2_reward = self.env.calculate_built_in_reward(1)
                        p2_episode_reward += p2_reward
                        p2_next_state = self.env.get_state(1)
                        self.memory_rl.add(p2_state, p2_action, p2_reward, p2_next_state, done)
                    p1_reward = self.env.calculate_built_in_reward(0)
                    p1_episode_reward += p1_reward
                    self.wins += 1
                    self.memory_rl.add(p1_state, p1_action, p1_reward, p1_next_state, done)
                else:
                    p1_reward = self.env.calculate_built_in_reward(0)
                    p1_episode_reward += p1_reward
                    self.memory_rl.add(p1_state, p1_action, p1_reward, p1_next_state, done)
                if self.steps % self.batch_size == 0:
                    if train:
                        self.train()
                    else:
                        return

            p2_state = self.env.get_state(1)
            while self.env.current_player_index == 1 and not done:
                p2_action = self.get_action(self.env.get_legal_actions(), p2_state)
                self.steps += 1
                p2_next_state, done = self.env.step(p2_action)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                if self.env.current_player_index == 0:
                    p1_reward = self.env.calculate_built_in_reward(0)
                    p1_episode_reward += p1_reward
                    p1_next_state = self.env.get_state(0)
                    self.memory_rl.add(p1_state, p1_action, p1_reward, p1_next_state, done)
                elif done:
                    p1_reward = self.env.calculate_built_in_reward(0)
                    p1_episode_reward += p1_reward
                    p1_next_state = self.env.get_state(0)
                    self.memory_rl.add(p1_state, p1_action, p1_reward, p1_next_state, done)

                    p2_reward = self.env.calculate_built_in_reward(1)
                    p2_episode_reward += p2_reward
                    self.memory_rl.add(p2_state, p2_action, p2_reward, p2_next_state, done)
                    self.p2.wins += 1
                else:
                    p2_reward = self.env.calculate_built_in_reward(1)
                    p2_episode_reward += p2_reward
                    self.memory_rl.add(p2_state, p2_action, p2_reward, p2_next_state, done)

                if self.steps % self.batch_size == 0:
                    if train:
                        self.train()
                    else:
                        return

        self.total_rewards_p1.append(p1_episode_reward)
        self.total_rewards_p2.append(p2_episode_reward)
        return p1_episode_reward, p2_episode_reward

    def train_agent(self):
        with tqdm(total=self.number_of_episodes) as progress_bar:
            progress_bar.update(1)
            while self.rounds < self.number_of_episodes:
                with warnings.catch_warnings():  # Mean of empty array
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    progress_bar.set_description("Agent 1 wins: " + str(self.wins)
                                                 + " Agent 2 wins: " + str(self.p2.wins)
                                                 + " Current epsilon: " + str(self.epsilon)
                                                 + " Average loss: " + str(np.average(self.losses)))
                self.play()
                self.rounds += 1
                progress_bar.update(1)
                if self.rounds % self.model_save_frequency == 0:
                    utils.plot(self.total_rewards_p1, self.losses, "DQN", rewards2=self.total_rewards_p2)
                    self.save()

        utils.plot(self.total_rewards_p1, self.losses, "DQN", rewards2=self.total_rewards_p2)
        self.save()

    def save(self):
        if not self.save_model:
            return
        folder = f'models/DQN/{utils.get_timestamp()}'
        print("Saving in file: " + folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.policy_network.save(f'{folder}/policy_model_{self.n_batches}.h5')
        self.supervised_learning_network.save(f'{folder}/supervised_model_{self.n_batches}.h5')

    def reinforcement_learning(self, fit=True):
        states, actions, rewards, next_states, dones, importance, indices = \
            self.memory_rl.sample(self.batch_size, alpha=self.alpha, beta=(1 - self.epsilon))

        q_values = self.policy_network.predict(states)

        old_q_values = q_values.copy()

        max_future_q = np.max(self.target_network.predict(next_states), axis=1)

        for i in range(len(indices)):
            q_values[i, actions[i]] = rewards[i]
            if not dones[i]:
                q_values[i, actions[i]] += self.gamma * max_future_q[i]
            q_values[i, actions[i]] = self.clip(q_values[i, actions[i]])

        td_errors = abs(q_values - old_q_values)
        td_errors = np.take_along_axis(td_errors, actions[:, None], axis=1)

        if not fit:
            return states, q_values, importance
        history = self.policy_network.fit(x=states, y=q_values, batch_size=self.batch_size, verbose=0,
                                          callbacks=[self.reduce_lr], sample_weight=importance)
        self.memory_rl.set_priorities(indices, td_errors)
        self.losses.append(history.history['loss'][0])

    def supervised_learning(self):
        states, actions = self.memory_sl.sample_batch(self.batch_size)
        self.supervised_learning_network.fit(x=states, y=actions, batch_size=self.batch_size, verbose=0,
                                             callbacks=[self.reduce_lr])

    def clip(self, value):
        if value >= 1:
            return 1
        if value <= -1:
            return -1
        return value

if __name__ == '__main__':
    agent = DeepQNetworkAgent()
    agent.train_agent()
