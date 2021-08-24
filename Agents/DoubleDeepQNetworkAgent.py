import os
import numpy as np
from Agents.DeepQNetworkAgent import DeepQNetworkAgent
from Utils import utils


class DoubleQNetworkAgent(DeepQNetworkAgent):
    def __init__(self):
        super().__init__(alpha=0.043763, delta=0.21649, epsilon_decay=0.99996, eta=0.01481,
                         gamma=0.029922, learning_rate=0.03727)

    def reinforcement_learning(self, fit=True):
        states, actions, rewards, next_states, dones, importance, indices = \
            self.memory_rl.sample(self.batch_size, alpha=self.alpha, beta=(1 - self.epsilon))

        q_values = self.policy_network.predict(states)
        old_q_values = q_values.copy()

        # You need to get max actions here, which means index, not it's value (so remake this part)
        predicted_action_values = self.policy_network.predict(next_states)
        # Prediction network determines best actions to take (get actions here)
        best_actions = np.argmax(predicted_action_values, axis=1)
        # But their values are determined by the target network (get their values here)
        intermediate_value = self.target_network.predict(next_states)
        # Takes values by index (action) in each row (take values of best actions)
        max_future_q = np.take_along_axis(intermediate_value, best_actions[:, None], axis=1)
        max_future_q = max_future_q.flatten()

        for i in range(len(indices)):
            # transition: state, action, reward, new_state, done
            # Calculate expected Q values
            q_values[i, actions[i]] = rewards[i]
            if not dones[i]:
                q_values[i, actions[i]] += self.gamma * max_future_q[i]

        td_errors = abs(q_values - old_q_values)
        td_errors = np.take_along_axis(td_errors, actions[:, None], axis=1)

        if not fit:
            return states, q_values, importance

        # Fit on all samples as one batch
        # Train Q network only, target network remains fixed
        history = self.policy_network.fit(x=states, y=q_values, batch_size=self.batch_size, verbose=0,
                                          callbacks=[self.reduce_lr], sample_weight=importance)
        self.memory_rl.set_priorities(indices, td_errors)
        self.losses.append(history.history['loss'][0])

        if self.n_batches % self.model_update_frequency == 0:
            # Update target network with weight of main network
            # Copy Q network to target network
            self.target_network.set_weights(self.policy_network.get_weights())

    def save(self):
        if not self.save_model:
            return
        folder = f'models/DoubleDQN/{utils.get_timestamp()}'
        print("Saving in file: " + folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.policy_network.save(f'{folder}/policy_model_{self.n_batches}.h5')
        self.supervised_learning_network.save(f'{folder}/supervised_model_{self.n_batches}.h5')


if __name__ == '__main__':
    agent = DoubleQNetworkAgent()
    agent.train_agent()
