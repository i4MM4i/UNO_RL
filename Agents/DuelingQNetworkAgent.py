import os

import numpy as np
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras as keras

from Agents.DeepQNetworkAgent import DeepQNetworkAgent
from Agents.DuelingQNetwork import DuelingQNetwork
from Utils import utils
from tensorflow.keras.losses import Huber, Reduction


class DuelingQNetworkAgent(DeepQNetworkAgent):
    def __init__(self):
        super().__init__()
        self.policy_network = DuelingQNetwork(self.action_size)
        self.target_network = DuelingQNetwork(self.action_size)
        self.policy_network.compile(optimizer=keras.optimizers.Adam(), loss=Huber(reduction=Reduction.SUM))
        self.update_network()

    def update_network(self):
        for t, e in zip(self.target_network.trainable_variables, self.policy_network.trainable_variables):
            t.assign(e)

    def reinforcement_learning(self, fit=True):
        states, actions, rewards, next_states, dones, importance, indices = \
            self.memory_rl.sample(self.batch_size, alpha=self.alpha, beta=(1 - self.epsilon))

        # Predict Q(s,a) given the batch of states
        q_values = self.policy_network.predict(states)
        old_q_values = q_values.copy()
        # Predict Q(s',a') from the evaluation network
        q_values_next = self.policy_network.predict(next_states)

        # extract the best action from the next state
        best_actions = np.argmax(q_values_next, axis=1)

        # get all the q values from the next state
        q_from_target = self.target_network(next_states).numpy()

        for i in range(len(indices)):
            # transition: state, action, reward, new_state, done
            # Calculate expected Q values
            q_values[i, actions[i]] = rewards[i]
            if not dones[i]:
                # add the discounted estimated reward from the selected best action (q_values_nxt)
                q_values[i, actions[i]] += self.gamma * q_from_target[i, best_actions[i]]

        if not fit:
            return states, q_values_next, importance
        td_errors = abs(q_values - old_q_values)
        td_errors = np.take_along_axis(td_errors, actions[:, None], axis=1)
        loss = self.policy_network.train_on_batch(states, q_values)
        self.memory_rl.set_priorities(indices, td_errors)
        self.losses.append(loss)

    def save(self):
        folder = f'models/DuelingDQN/{utils.get_timestamp()}'
        print("Saving in file: " + folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.policy_network.save(f'{folder}', save_format='tf')


if __name__ == '__main__':
    agent = DuelingQNetworkAgent()
    agent.train_agent()
