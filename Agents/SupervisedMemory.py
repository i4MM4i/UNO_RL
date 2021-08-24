from collections import deque
import numpy as np


class SupervisedMemory:
    def __init__(self, experience_size):
        self.states = deque(maxlen=experience_size)
        self.actions = deque(maxlen=experience_size)

    def __len__(self):
        return len(self.states)

    def add(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    def sample_batch(self, batch_size):
        assert batch_size < len(self.states)
        indices = np.random.choice(np.arange(len(self.states)), batch_size, replace=False).astype(int)
        sample_states = np.array(self.states)[indices]
        sample_actions = np.array(self.actions)[indices]
        return sample_states, sample_actions




