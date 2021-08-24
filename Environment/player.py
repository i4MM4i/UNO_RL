from Environment.card import Card
from abc import ABCMeta, abstractmethod


class Player(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name
        self.cards = []
        self.wins = 0

    def print_hand(self):
        Card.pretty_print_cards(self.cards, True)

    def play_card(self, index):
        card_to_play = self.cards[index]
        del self.cards[index]
        return card_to_play

    def draw_cards(self, cards):
        self.cards += cards

    def get_hand_size(self):
        return len(self.cards)

    def get_drawn_card(self):
        return self.cards(self.get_hand_size()-1)

    @abstractmethod
    def get_action(self):
        pass

    def reset(self):
        self.cards = []
