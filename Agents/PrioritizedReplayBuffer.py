from collections import deque

import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self, replay_buffer_size):
        self.states = deque(maxlen=replay_buffer_size)
        self.actions = deque(maxlen=replay_buffer_size)
        self.rewards = deque(maxlen=replay_buffer_size)
        self.next_states = deque(maxlen=replay_buffer_size)
        self.dones = deque(maxlen=replay_buffer_size)
        self.priorities = deque(maxlen=replay_buffer_size)

    def __len__(self):
        return len(self.states)

    def add(self,  state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, alpha):
        scaled_priorities = np.array(self.priorities, dtype=object) ** alpha
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities, beta):
        importance = (1 / len(self.states) * 1 / probabilities) ** beta
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, alpha=1.0, beta=1.0):
        sample_size = min(len(self.states), batch_size)
        sample_probabilities = self.get_probabilities(alpha)
        indices = random.choices(range(len(self.states)), k=sample_size, weights=sample_probabilities)
        sample_states = np.array(self.states)[indices].astype(np.short)
        sample_actions = np.array(self.actions)[indices].astype(np.uint8)
        sample_rewards = np.array(self.rewards)[indices].astype(float)
        sample_next_states = np.array(self.next_states)[indices].astype(np.short)
        sample_dones = np.array(self.dones)[indices].astype(bool)
        importance = self.get_importance(sample_probabilities[indices], beta).astype(float)
        return sample_states, sample_actions, sample_rewards, sample_next_states, sample_dones, importance, indices

    def set_priorities(self, indices, errors, offset=0.1):
        for index, error in zip(indices, errors):
            self.priorities[index] = abs(error) + offset
