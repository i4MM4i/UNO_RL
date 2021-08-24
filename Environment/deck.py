from random import shuffle
from Environment.card import Card


class Deck:

    def __init__(self):
        self.cards = self.generate_deck()
        self.shuffle_deck()

    def generate_deck(self):
        cards = self.generate_number_cards_for_each_color()

        cards += self.generate_action_card_for_each_color()

        cards += self.generate_wild_cards()

        return cards

    def generate_number_cards_for_each_color(self) -> [Card.Color]:
        """
        Generira karte od 0-9 in 1-9 za vse barve
        """
        number_cards = []

        for color in Card.Color:
            if color != Card.Color.BLACK:
                number_cards += [Card(color,
                                      Card.Kind.NUMBER,
                                      0,
                                      self.generate_action_number(color, 0), 0)]
                number_cards += [Card(color,
                                      Card.Kind.NUMBER,
                                      number,
                                      self.generate_action_number(color, number), number)
                                 for number in range(1, 10)] * 2
        return number_cards

    def generate_action_card_for_each_color(self, ):
        action_cards = []
        for repeat in range(2):
            for action in Card.ActionTraits:
                for color in Card.Color:
                    if color != Card.Color.BLACK:
                        action_cards += [Card(color,
                                              Card.Kind.ACTION,
                                              action,
                                              self.generate_action_number(color, action.value), 20)]
        return action_cards

    @staticmethod
    def generate_action_number(color, trait):
        number_of_cards_for_a_color = 13
        return trait + number_of_cards_for_a_color * color.value

    @staticmethod
    def generate_wild_cards():
        wild_cards = []
        for repeat in range(4):
            for wild_trait in Card.WildTraits:
                action_number = 0
                if wild_trait == 13:
                    action_number = 52
                if wild_trait == 14:
                    action_number = 53

                wild_cards += [Card(Card.Color.BLACK, Card.Kind.WILD, wild_trait, action_number, 50)]
        return wild_cards

    def shuffle_deck(self):
        shuffle(self.cards)

    def draw_cards(self, number):
        drawn_cards = self.cards[-number:]
        del self.cards[-number:]
        return drawn_cards

    def pretty_print_deck(self):
        Card.pretty_print_cards(self.cards, False)

    def count_num_for_a_color(self, color):
        count = 0
        for card in self.cards:
            if card.color == color:
                count += 1
        print(count)

    def get_number_of_cards_in_deck(self):
        return len(self.cards)

    def set_cards(self, cards):
        self.cards = cards

    def reset(self):
        self.__init__()
