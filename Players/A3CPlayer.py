import numpy as np
import tensorflow as tf

from Agents.ActorCriticModel import ActorCriticModel
from Environment.player import Player
from Environment.state import State


class A3CPlayer(Player):
    def __init__(self, model_path):
        super().__init__("A3C Player")
        self.action_size = State.ACTION_SIZE
        self.state_size = State.STATE_SIZE
        self.model = ActorCriticModel(self.action_size)
        self.model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        self.model.load_weights(model_path)

    def get_action(self, legal_actions, state):
        policy, value = self.model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        mask = np.full(self.action_size, True)
        mask[legal_actions] = False
        policy = np.ma.array(policy, mask=mask)
        action = np.argmax(policy)
        return action
