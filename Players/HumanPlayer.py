"""
Placeholder
"""
from Environment.player import Player


class HumanPlayer(Player):
    def __init__(self):
        super().__init__("Human")

    def get_action(self, legal_actions, state):
        raise NotImplementedError

