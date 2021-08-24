import threading

import numpy as np
from collections import Counter

from Environment.card import Card


class State:
    STATE_SIZE = 229 #277 #377
    ACTION_SIZE = 55

    def encode_state(self, hand, opponents_hand_size, played_cards, top_card):
        encoded_hand = self.encode_visible_cards(hand)  # 108
        encoded_played_cards = self.encode_visible_cards(played_cards)  # 108
        encoded_opponents_hand = self.bin_array(opponents_hand_size, 7) # self.encode_hidden_cards(opponents_hand_size)  # 7 (prej 107)
        encoded_top = self.bin_array(top_card.action_number, 6)#self.encode_top_card(top_card)  # 53

        state = []
        state.extend(encoded_hand)
        state.extend(encoded_played_cards)
        state.extend(encoded_opponents_hand)
        state.extend(encoded_top)
        state = np.array(state)
        state = state.astype(int)
        return state

    def encode_visible_cards(self, cards):
        encoded_cards = np.zeros(108)
        card: Card
        counted_cards = self.count_cards(cards)
        for i, card_with_count in enumerate(counted_cards):
            card, count = self.split_to_card_and_count(card_with_count)
            if card.trait == 0:
                index = card.color
            elif card.kind == Card.Kind.WILD:
                if card.trait == Card.WildTraits.WILD:
                    index = 99 + count
                elif card.trait == Card.WildTraits.DRAW_4:
                    index = 103 + count
            else:
                index = (card.trait * 2 - (2 - count)) * 4 + card.color + 1
            encoded_cards[index] = 1
        return encoded_cards

    @staticmethod
    def encode_hidden_cards(number_of_cards):
        encoded_hidden_cards = np.zeros(107)
        for index in range(number_of_cards):
            encoded_hidden_cards[index] = 1
        return encoded_hidden_cards

    @staticmethod
    def encode_top_card(card):
        encoded_top = np.zeros(54)
        encoded_top[card.action_number] = 1
        return encoded_top

    @staticmethod
    def split_to_card_and_count(card_with_count):
        return Card(card_with_count[0][0], card_with_count[0][1], card_with_count[0][2], card_with_count[0][3],
                    card_with_count[0][4]), card_with_count[1]

    @staticmethod
    def count_cards(cards):
        card: Card
        return Counter((card.color, card.kind, card.trait, card.action_number, card.points) for card in cards) \
            .most_common()

    @staticmethod
    def bin_array(num: int, m):
        """Converts a positive integer into an m-bit numpy array"""
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)



"""
class AbstractState:

    @staticmethod
    def encode_state(top, hand, legal_actions):
        hand = AbstractState.encode_hand(hand, legal_actions)
        #print(AbstractState.encode_hand(hand, legal_actions))
        top = AbstractState.encode_top_color(top)
        #print(AbstractState.encode_top_color(top))
        state = np.concatenate((top, hand))
        state = state
        return state

    @staticmethod
    def bin_array(num: int, m):
        ""Converts a positive integer into an m-bit numpy array""
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

    @staticmethod
    def encode_top_color(card):
        return AbstractState.bin_array(card.color.value, 2)

    @staticmethod
    def encode_hand(hand, legal_actions):
        ""Number of number cards in hand""
        red = 0
        playable_red = 0
        yellow = 0
        playable_yellow = 0
        green = 0
        playable_green = 0
        blue = 0
        playable_blue = 0

        skip = 0
        playable_skip = 0
        reverse = 0
        playable_reverse = 0
        draw_2 = 0
        playable_draw_2 = 0
        wild = 0
        playable_wild = 0
        draw_4 = 0
        playable_draw_4 = 0
        for card in hand:
            if card.kind == Card.Kind.NUMBER:
                if card.color == Card.Color.RED:
                    red += 1
                    if card.action_number in legal_actions:
                        playable_red += 1
                elif card.color == Card.Color.YELLOW:
                    yellow += 1
                    if card.action_number in legal_actions:
                        playable_yellow += 1
                elif card.color == Card.Color.GREEN:
                    green += 1
                    if card.action_number in legal_actions:
                        playable_green += 1
                else:
                    blue += 1
                    if card.action_number in legal_actions:
                        playable_blue += 1
            elif card.kind == Card.Kind.ACTION:
                if card.trait == Card.ActionTraits.SKIP:
                    skip += 1
                    if card.action_number in legal_actions:
                        playable_skip += 1
                if card.trait == Card.ActionTraits.REVERSE:
                    reverse += 1
                    if card.action_number in legal_actions:
                        playable_reverse += 1
                else:
                    draw_2 += 1
                    if card.action_number in legal_actions:
                        playable_draw_2 += 1
            elif card.kind == Card.Kind.WILD:
                if card.trait == Card.WildTraits.WILD:
                    wild += 1
                    if card.action_number in legal_actions:
                        playable_wild += 1
                else:
                    draw_4 += 1
                    if card.action_number in legal_actions:
                        playable_draw_4 += 1
        red = min(red, 2)
        playable_red = min(playable_red, 2)
        yellow = min(yellow, 2)
        playable_yellow = min(playable_yellow, 2)
        green = min(green, 2)
        playable_green = min(playable_green, 2)
        blue = min(blue, 2)
        playable_blue = min(playable_blue, 2)

        skip = min(skip, 2)
        playable_skip = min(playable_skip, 2)
        reverse = min(reverse, 2)
        playable_reverse = min(playable_reverse, 2)
        draw_2 = min(draw_2, 2)
        playable_draw_2 = min(playable_draw_2, 2)
        wild = min(wild, 2)
        playable_wild = min(playable_wild, 2)
        draw_4 = min(draw_4, 2)
        playable_draw_4 = min(playable_draw_4, 2)
        args = (AbstractState.bin_array(red, 2),
                AbstractState.bin_array(yellow, 2),
                AbstractState.bin_array(green, 2),
                AbstractState.bin_array(blue, 2),
                AbstractState.bin_array(playable_red, 2),
                AbstractState.bin_array(playable_yellow, 2),
                AbstractState.bin_array(playable_green, 2),
                AbstractState.bin_array(playable_blue, 2),
                AbstractState.bin_array(skip, 2),
                AbstractState.bin_array(reverse, 2),
                AbstractState.bin_array(draw_2, 2),
                AbstractState.bin_array(wild, 2),
                AbstractState.bin_array(draw_4, 2),
                AbstractState.bin_array(playable_skip, 2),
                AbstractState.bin_array(playable_reverse, 2),
                AbstractState.bin_array(playable_draw_2, 2),
                AbstractState.bin_array(playable_wild, 2),
                AbstractState.bin_array(playable_draw_4, 2)
                )
        return np.concatenate(args)"""


