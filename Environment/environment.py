import threading

import numpy as np
from Environment.player import Player
from Environment.deck import Deck
from Environment.card import Card
from Environment.state import State #, AbstractState


class UnoEnvironment:
    def __init__(self, print_state=False):
        self.number_of_players = 2
        self.print_state = print_state
        self.deck = Deck()
        self.state_tracker = State()
        self.players = None
        self.played_cards = None
        self.top = None
        self.played_card = None
        self.current_player_index = None
        self.current_player = None
        self.previous_player_index = None
        self.turn = None
        self.done = None

        self.reward_info = [[None, None, None, None], [None, None, None, None]]
        #  if self.print_state:
        #  self.pretty_print_state()

    def reset(self, player_1: Player, player_2: Player):
        player_1.reset()
        player_2.reset()
        self.players = [player_1, player_2]
        self.deck.reset()
        self.players[0].draw_cards(self.deck.draw_cards(7))
        self.players[1].draw_cards(self.deck.draw_cards(7))
        self.played_card = None
        self.played_cards = self.reveal_top_card()
        self.top = self.played_cards[-1]
        self.current_player_index = 0
        self.current_player = self.players[0]
        self.previous_player_index = None
        self.turn = 0
        self.state_tracker = State()
        self.done = False
        if self.print_state:
            self.pretty_print_state()
        # TODO: use built in state function
        return self.get_state(0)
        #current_players_hand = self.current_player.cards
        # n_cards_other_players_hand = len(self.next_player().cards)
        #return self.state_tracker.encode_state(current_players_hand, n_cards_other_players_hand, self.played_cards,
                                               #self.top)

    def reveal_top_card(self):
        revealed_cards = self.deck.draw_cards(1)
        while revealed_cards[-1].kind != Card.Kind.NUMBER:
            revealed_cards += self.deck.draw_cards(1)

        return revealed_cards

    def check_win_or_continue(self, skip):
        #print("Should Skip", skip)
        self.turn += 1
        if len(self.players[0].cards) == 0:
            self.done = True
            if self.print_state:
                self.pretty_print_state()
                print(self.players[0].name + " wins!")
        elif len(self.players[1].cards) == 0:
            self.done = True
            if self.print_state:
                self.pretty_print_state()
                print(self.players[1].name + " wins!")
        else:
            self.next_turn(skip)
            """if skip:
                self.previous_player_index = self.current_player_index
                if self.print_state:
                    self.pretty_print_state()
            else:
                self.next_turn()"""

    def set_hand_info_before(self):
        if self.current_player_index == 0:
            # Moj hand ob začetku actiona - p1b
            self.reward_info[0][0] = self.players[0].get_hand_size()
            # Opponentov hand ob začetku actiona - p1ob
            self.reward_info[0][1] = self.players[1].get_hand_size()
        else:
            # Moj hand ob začetku actiona - p2b
            self.reward_info[1][0] = self.players[1].get_hand_size()
            # Opponentov hand ob začetku actiona - p2ob
            self.reward_info[1][1] = self.players[0].get_hand_size()

    def set_hand_info_after(self):
        # Po koncu opponentovega actiona sm js na vrsti spet al pa on, če se je zgodil skip
        # Loh sm tud js na vrsti takoj za sabo če skippam
        if self.current_player_index == 0:
            # Moj hand ob koncu opponentovega actiona - p2a
            self.reward_info[0][2] = self.players[0].get_hand_size()
            # Opponentov hand ob koncu opponentovega actiona - p2oa
            self.reward_info[0][3] = self.players[1].get_hand_size()
        else:
            # Moj hand ob koncu opponentovega actiona - p1a
            self.reward_info[1][2] = self.players[1].get_hand_size()
            # Opponentov hand ob koncu opponentovega actiona - p1oa
            self.reward_info[1][3] = self.players[0].get_hand_size()

    def step(self, action):

        self.set_hand_info_before() # Current player je ta k izvaja step
        self.take_action(action)
        self.set_hand_info_after() # Current player je nasledni, k je na vrsti

        #current_players_hand = self.players[self.current_player_index].cards
        #opponents_hand_len = len(self.players[self.next_player_index()].cards)
        #self.top = self.played_cards[-1]

        #print("Reward info:", self.reward_info)
        return self.get_state(self.previous_player_index), self.done

        #return self.state_tracker.encode_state(current_players_hand, opponents_hand_len, self.played_cards,
                                               #self.top), self.done

    def step_with_opp_step(self, action):
        #if len(self.deck.cards) + len(self.players[0].cards) + len(self.players[1].cards) \
         #       + len(self.played_cards) != 108:
          #  print("Fuck", len(self.deck.cards) + len(self.players[0].cards) + len(self.players[1].cards)
           #       + len(self.played_cards))
        self.set_hand_info_before()  # Current player je ta k izvaja step
        self.take_action(action)
        self.set_hand_info_after()  # Current player je nasledni, k je na vrsti
        # Opponent is part of the environment, so it has to make it's actions before returning the new state and rewards
        self.opponent_action()

        reward = self.calculate_built_in_reward(0)

        #agents_hand = self.players[0].cards
        #opponents_hand_len = len(self.players[1].cards)
        #self.top = self.played_cards[-1]
        #if len(self.deck.cards) + len(self.players[0].cards) + len(self.players[1].cards) \
         #       + len(self.played_cards) != 108:
          #  print("Fuck", len(self.deck.cards) + len(self.players[0].cards) + len(self.players[1].cards)
           #       + len(self.played_cards))
        #state = self.state_tracker.encode_state(agents_hand, opponents_hand_len, self.played_cards, self.top)
        return self.get_state(0), reward, self.done

    def opponent_action(self):
        while self.current_player_index == 1 and not self.done:
            self.set_hand_info_before()  # Current player je ta k izvaja step
            action = self.players[1].get_action(self.get_legal_actions(), self.get_state(1))
            self.take_action(action)
            self.set_hand_info_after()  # Current player je nasledni, k je na vrsti

    def take_action(self, action):
        skip = False
        if action != 54:
            # Get appropriate card from players hand depending on provided action
            index = [card.action_number for card in self.current_player.cards].index(action)
            self.played_card = self.current_player.cards[index]
        else:
            self.played_card = None

        if not self.played_card:
            self.draw_cards(self.current_player, 1)
            drawn_card_action_number = self.current_player.cards[len(self.current_player.cards)-1].action_number
            if drawn_card_action_number in self.get_legal_actions():
                # If drawn card is legal to play, then do so
                if self.print_state:
                    print("Automatically playing drawn card!")
                self.take_action(drawn_card_action_number)
                return
        else:
            skip = self.play_card(index)
            """if self.played_card.trait == Card.ActionTraits.SKIP or self.played_card.trait == Card.ActionTraits.REVERSE:
                skip = True
            elif self.played_card.trait == Card.ActionTraits.DRAW_2:
                self.draw_cards(self.next_player(), 2)
                skip = True
            elif self.played_card.trait == Card.WildTraits.WILD:
                self.played_card.color = Card.Color(np.random.randint(3))
            elif self.played_card.trait == Card.WildTraits.DRAW_4:
                self.draw_cards(self.next_player(), 4)
                self.played_card.color = Card.Color(np.random.randint(3))
                skip = True

            self.played_cards.append(self.current_player.play_card(index))"""
        self.check_win_or_continue(skip)

    def play_card(self, index):
        skip = False
        if self.played_card.trait == Card.ActionTraits.SKIP or self.played_card.trait == Card.ActionTraits.REVERSE:
            skip = True
        elif self.played_card.trait == Card.ActionTraits.DRAW_2:
            self.draw_cards(self.next_player(), 2)
            skip = True
        elif self.played_card.trait == Card.WildTraits.WILD:
            self.played_card.color = Card.Color(np.random.randint(3))
        elif self.played_card.trait == Card.WildTraits.DRAW_4:
            self.draw_cards(self.next_player(), 4)
            self.played_card.color = Card.Color(np.random.randint(3))
            skip = True

        self.played_cards.append(self.current_player.play_card(index))
        return skip

    '''def calculate_reward(self, hand_size_before, opponents_hand_size_before,
                         hand_size_after, opponents_hand_size_after, player_index):
        if self.done:
            if self.players[player_index].get_hand_size() == 0:
                reward = 1
            else:
                reward = -1
        else:
            if self.current_player_index == self.previous_player_index:
                # Skipping is always positive
                return 1
            elif hand_size_before > hand_size_after < opponents_hand_size_after:
                # Decreasing hand size is positive, unless you're behind
                return 1
            return -1
            if self.current_player_index == self.previous_player_index \
                    or hand_size_after < hand_size_before \
                    or opponents_hand_size_after > opponents_hand_size_before:
                # Means there was a skip
                return 1
            else:
                return -1
        return reward'''

    def calculate_built_in_reward(self, player_index):
        if self.done:
            if self.players[player_index].get_hand_size() == 0:
                reward = 1
            else:
                reward = -1
        else:
            hand_size_before = self.reward_info[player_index][0]
            hand_size_after = self.reward_info[player_index][2]
            opponents_hand_size_after = self.reward_info[player_index][3]
            #if self.current_player_index == self.previous_player_index or \
            if hand_size_before > hand_size_after < opponents_hand_size_after:
                # Skipping is always positive
                # Decreasing hand size is positive, unless you're behind
                return 1
            #elif hand_size_before > hand_size_after < opponents_hand_size_after:
                # Decreasing hand size is positive, unless you're behind
                #return 1
            return -1
        return reward

    '''def clip_value(self, value):
        if value > 0:
            value = 0
        elif value <= 0:
            value = -1
        return value'''

    def get_state(self, player_index):
        current_hand = self.players[player_index].cards
        opponents_hand_len = len(self.players[(player_index + 1) % self.number_of_players].cards)
        self.top = self.played_cards[-1]
        return self.state_tracker.encode_state(current_hand, opponents_hand_len, self.played_cards,
                                               self.top)

    """def get_abstract_state(self):
        self.top = self.played_cards[-1]
        return AbstractState.encode_state(self.top, self.players[0].cards, self.get_legal_actions())"""

    def get_legal_actions(self):
        self.top = self.played_cards[-1]
        legal_actions = []
        draw_4_actions = []
        matches_top_color = False
        if self.top.kind == Card.Kind.WILD:
            for card in self.current_player.cards:
                if card.kind == Card.Kind.WILD:
                    if card.trait == Card.WildTraits.DRAW_4:
                        draw_4_actions = card.action_number
                    else:
                        legal_actions.append(card.action_number)
                if card.color == self.top.color:
                    matches_top_color = True
                    legal_actions.append(card.action_number)

        else:
            for card in self.current_player.cards:
                if card.kind == Card.Kind.WILD:
                    if card.trait == Card.WildTraits.DRAW_4:
                        draw_4_actions = card.action_number
                    else:
                        legal_actions.append(card.action_number)
                if card.color == self.top.color:
                    matches_top_color = True
                    legal_actions.append(card.action_number)
                elif card.trait == self.top.trait:
                    legal_actions.append(card.action_number)

        # Only if there are no other card with same color as the top actions except draw, draw_4 is legal
        if not matches_top_color and draw_4_actions:
            legal_actions.append(draw_4_actions)
        legal_actions.append(54)
        return legal_actions

    def get_playable_cards_for_current_player(self):
        legal_actions = self.get_legal_actions()
        playable_cards = []
        for card in self.current_player.cards:
            if card.action_number in legal_actions and card not in playable_cards:
                playable_cards.append(card)
        return playable_cards

    def next_player(self):
        return self.players[self.next_player_index()]

    def next_player_index(self):
        return (self.current_player_index + 1) % self.number_of_players

    def draw_cards(self, player, number_of_cards_to_draw):
        # Handle empty deck
        cards_to_draw = []
        for draw in range(number_of_cards_to_draw):
            if self.deck.get_number_of_cards_in_deck() == 0:
                self.shuffle_played_cards_into_deck()
            cards_to_draw += self.deck.draw_cards(1)
        player.draw_cards(cards_to_draw)

    def shuffle_played_cards_into_deck(self):
        if len(self.played_cards) == 1:
            # Trying to draw from empty deck and only one card left on the played pile
            return
        temp = [self.played_cards.pop(-1)]
        for card in self.played_cards:
            # reset randomly assigned color
            if card.trait == Card.WildTraits.WILD or card.trait == Card.WildTraits.DRAW_4:
                card.color = Card.Color.BLACK

        self.deck.set_cards(self.played_cards)
        self.played_cards = temp
        self.deck.shuffle_deck()

    def get_hand(self):
        return self.current_player.cards

    def get_other_hand_size(self):
        return len(self.players[self.next_player_index()].cards)

    def next_turn(self, skip):
        if skip:
            self.previous_player_index = self.current_player_index
        else:
            self.current_player = self.next_player()  # cycles 0 and 1
            self.previous_player_index = self.current_player_index
            self.current_player_index = self.next_player_index()

        if self.print_state:
            self.pretty_print_state()

    def pretty_print_state(self):
        print("-------------------------------------------------------------------------------------------------------")
        print("Turn " + str(self.turn))
        Card.pretty_print_cards(self.played_cards[len(self.played_cards) - 1:], False)

        for i, player in enumerate(self.players):

            if i == self.current_player_index:
                print("ACTIVE-", end='')
            print(player.name + "'s hand:")
            player.print_hand()
        print("-------------------------------------------------------------------------------------------------------")
