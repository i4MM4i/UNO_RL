import numpy as np

from Environment.player import Player


class RandomPlayer(Player):
    def __init__(self):
        super().__init__("Random Player")

    def get_action(self, legal_actions, state):
        return np.random.choice(legal_actions, 1)
