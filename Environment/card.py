from dataclasses import dataclass
from termcolor import colored
from enum import IntEnum, Enum


@dataclass
class Card:
    color: int  # defines color
    kind: int  # defines card type (number, action, wild)
    trait: int  # defines trait aka number
    action_number: int  # defines action number for ML
    points: int  # defines points for rewards

    class Color(IntEnum):
        RED = 0
        YELLOW = 1
        GREEN = 2
        BLUE = 3
        BLACK = 4

    class Kind(IntEnum):
        NUMBER = 0
        ACTION = 1
        WILD = 2

    class ActionTraits(IntEnum):
        SKIP = 10
        REVERSE = 11
        DRAW_2 = 12

    class WildTraits(IntEnum):
        WILD = 13
        DRAW_4 = 14

    # Think through if something similar can be used to simplify move legality checking, USELESS FOR NOW
    def can_be_played(self, top_card):
        if (self.color == top_card.color) or (self.trait == self.trait) or (self.kind == Card.Kind.WILD):
            return True
        return False

    def is_same_color(self, other):
        return self.color == other.color

    def is_same_number(self, other):
        return self.number == other.number

    @staticmethod
    def pretty_print_cards(cards, print_id):
        if print_id:
            print('0:'+colored('Draw', 'cyan'), end=' ')
        for index, card in enumerate(cards):
            if print_id:
                print(str(index+1)+':', end='')
            if card.trait > 9:
                if card.color == Card.Color.RED:
                    print(colored(card.trait.name, 'red'), end=' ')
                elif card.color == Card.Color.YELLOW:
                    print(colored(card.trait.name, 'white'), end=' ')
                elif card.color == Card.Color.GREEN:
                    print(colored(card.trait.name, 'green'), end=' ')
                elif card.color == Card.Color.BLUE:
                    print(colored(card.trait.name, 'blue'), end=' ')
                elif card.color == Card.Color.BLACK:
                    print(colored(card.trait.name, 'magenta'), end=' ')
            else:
                if card.color == Card.Color.RED:
                    print(colored(card.trait, 'red'), end=' ')
                elif card.color == Card.Color.YELLOW:
                    print(colored(card.trait, 'white'), end=' ')
                elif card.color == Card.Color.GREEN:
                    print(colored(card.trait, 'green'), end=' ')
                elif card.color == Card.Color.BLUE:
                    print(colored(card.trait, 'blue'), end=' ')
                elif card.color == Card.Color.BLACK:
                    print(colored(card.trait, 'magenta'), end=' ')
        print('\n')





